# app.py

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from yomitoku import DocumentAnalyzer
import json

app = FastAPI()

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境ではより制限的に設定することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DocumentAnalyzerのインスタンスを作成
def create_analyzer(ignore_line_break: bool = False) -> DocumentAnalyzer:
    return DocumentAnalyzer(
        configs={
            "ocr": {
                "text_detector": {
                    "device": "cuda",
                    "visualize": False,
                },
                "text_recognizer": {
                    "device": "cuda",
                    "visualize": False,
                },
            },
            "layout_analyzer": {
                "layout_parser": {
                    "device": "cuda",
                    "visualize": False,
                },
                "table_structure_recognizer": {
                    "device": "cuda",
                    "visualize": False,
                },
            },
        },
        device="cuda",
        visualize=False,
        ignore_line_break=ignore_line_break
    )

# DocumentAnalyzerのインスタンスを作成
analyzer = DocumentAnalyzer(
    configs={
        "ocr": {
            "text_detector": {
                "device": "cuda",
                "visualize": False,
            },
            "text_recognizer": {
                "device": "cuda",
                "visualize": False,
            },
        },
        "layout_analyzer": {
            "layout_parser": {
                "device": "cuda",
                "visualize": False,
            },
            "table_structure_recognizer": {
                "device": "cuda",
                "visualize": False,
            },
        },
    },
    device="cuda",
    visualize=False
)

@app.post("/analyze")
async def analyze_document(
    file: UploadFile = File(...),
    format: str = Query("json", description="出力フォーマット (json, markdown, vertical, horizontal)")
):
    """
    画像からテキストを抽出するエンドポイント

    format:
        - json: 完全なJSONレスポンス
        - markdown: Markdown形式
        - vertical: 垂直方向のテキストのみ
        - horizontal: 水平方向のテキストのみ
    """
    # ファイルを読み込み
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 画像を解析
    results, _, _ = await analyzer(img)

    # フォーマットに応じて結果を返す
    format = format.lower()
    if format == "markdown":
        import tempfile
        import os

        # 一時ファイルを作成してmarkdownを生成
        with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
            results.to_markdown(temp_file.name)
            with open(temp_file.name, 'r', encoding='utf-8') as f:
                markdown_content = f.read()
            os.unlink(temp_file.name)  # 一時ファイルを削除

        return {"format": "markdown", "content": markdown_content}
    elif format == "vertical" or format == "horizontal":
        json_content = json.loads(results.model_dump_json())
        # 指定された方向のテキストのみを抽出して結合
        text = "\n".join([
            p["contents"] for p in json_content["paragraphs"]
            if p["direction"] == format and p["contents"] is not None
        ])
        return {"format": format, "content": text}
    else:  # json
        return {"format": "json", "content": json.loads(results.model_dump_json())}

        # レスポンスの作成
        return {
            "format": "json",
            "content": json_content,
            "vertical_only": vertical_text,
            "horizontal_only": horizontal_text
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
