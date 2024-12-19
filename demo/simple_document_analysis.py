import cv2
import torch

from yomitoku import DocumentAnalyzer
from yomitoku.data.functions import load_pdf

if __name__ == "__main__":
    # GPUメモリの使用状況を表示
    if torch.cuda.is_available():
        print(f"GPU Memory Usage before loading model:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

    # ONNXランタイムを有効にし、バッチサイズを小さくする設定
    configs = {
        "ocr": {
            "text_recognizer": {
                "device": "cuda",
                "visualize": True,
                "infer_onnx": True,  # ONNXランタイムを有効化
                "model_name": "parseq-small",  # liteモデルを使用
                "data": {
                    "batch_size": 32  # バッチサイズを小さくする
                }
            }
        }
    }

    analyzer = DocumentAnalyzer(configs=configs, visualize=True, device="cuda")

    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage after loading model:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

    # PDFファイルを読み込み
    imgs = load_pdf("demo/sample.pdf")
    for i, img in enumerate(imgs):
        results, ocr_vis, layout_vis = analyzer(img)

        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage during inference:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")

        # HTML形式で解析結果をエクスポート
        results.to_html(f"output_{i}.html")

        # 可視化画像を保存
        cv2.imwrite(f"output_ocr_{i}.jpg", ocr_vis)
        cv2.imwrite(f"output_layout_{i}.jpg", layout_vis)
