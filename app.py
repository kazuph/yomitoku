# app.py

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from yomitoku import DocumentAnalyzer
import json
import logging

# ロガーの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORSミドルウェアを追加
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境ではより制限的に設定することを推奨
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from yomitoku.data.functions import load_pdf

# DocumentAnalyzerのインスタンスを作成
logger.info("Starting to initialize DocumentAnalyzer...")
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
logger.info("DocumentAnalyzer initialization completed!")

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI server is ready to handle requests!")

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
    # ファイルを一時保存
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
        contents = await file.read()
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        # PDFかどうかを判定
        if file.filename.lower().endswith('.pdf'):
            imgs = load_pdf(temp_path)
        else:
            # 画像として読み込み
            nparr = np.frombuffer(contents, np.uint8)
            imgs = [cv2.imdecode(nparr, cv2.IMREAD_COLOR)]

        all_results = []
        for img in imgs:
            results, _, _ = await analyzer(img)
            all_results.append(results)

        # フォーマットに応じて結果を返す
        format = format.lower()
        if format == "markdown":
            # 一時ファイルを作成してmarkdownを生成
            markdown_contents = []
            for i, results in enumerate(all_results):
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as md_file:
                    results.to_markdown(md_file.name, img=imgs[i])
                    with open(md_file.name, 'r', encoding='utf-8') as f:
                        markdown_contents.append(f.read())
                    os.unlink(md_file.name)  # 一時ファイルを削除

            # ページ番号を追加して連結
            final_markdown = ""
            for i, content in enumerate(markdown_contents):
                if i > 0:
                    final_markdown += f"\n\n## Page {i + 1}\n\n"
                final_markdown += content

            return {"format": "markdown", "content": final_markdown}

        elif format == "vertical" or format == "horizontal":
            all_text = []
            for results in all_results:
                json_content = json.loads(results.model_dump_json())
                # 指定された方向のテキストのみを抽出して結合
                text = "\n".join([
                    p["contents"] for p in json_content["paragraphs"]
                    if p["direction"] == format and p["contents"] is not None
                ])
                all_text.append(text)
            return {"format": format, "content": "\n\n".join(all_text)}

        else:  # json
            json_results = [json.loads(results.model_dump_json()) for results in all_results]
            return {"format": "json", "content": json_results}

    finally:
        # 一時ファイルを削除
        os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
