import io
import os
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO

# =====================================================
# FastAPI
# =====================================================

app = FastAPI(title="SmartInspect Cloud")

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)

# =====================================================
# モデル読み込み
# =====================================================

MODEL_PATH = "../runs/classify/runs/classify/screw_classifier/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"モデルが見つかりません: {MODEL_PATH}"
    )

model = YOLO(MODEL_PATH)

# =====================================================
# クラス名変換
# =====================================================

LABEL_MAP = {
    "good": "良品",
    "thread_top": "ネジ山先端欠陥",
    "thread_side": "ネジ山側面欠陥",
    "scratch_head": "ネジ頭部傷",
    "scratch_neck": "ネジ首部傷",
    "manipulated_front": "加工異常"
}

# =====================================================
# デバッグ設定
# =====================================================

SAVE_DEBUG_IMAGE = True

DEBUG_DIR = BASE_DIR / "debug_output"

if SAVE_DEBUG_IMAGE:
    DEBUG_DIR.mkdir(exist_ok=True)

# =====================================================
# 画像前処理
# =====================================================

def normalize_environment_changes(
    pil_image: Image.Image,
) -> Image.Image:

    img_np = np.array(pil_image)

    if img_np.ndim == 2:

        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_GRAY2BGR
        )

    elif img_np.shape[2] == 4:

        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGBA2BGR
        )

    else:

        img_bgr = cv2.cvtColor(
            img_np,
            cv2.COLOR_RGB2BGR
        )

    gray = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2GRAY
    )

    target_mean = 128.0

    current_mean = np.mean(gray)

    if current_mean > 0:

        gray = cv2.addWeighted(
            gray,
            1.0,
            np.zeros_like(gray),
            0,
            target_mean - current_mean
        )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8)
    )

    normalized = clahe.apply(gray)

    denoised = cv2.bilateralFilter(
        normalized,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    processed_bgr = cv2.cvtColor(
        denoised,
        cv2.COLOR_GRAY2BGR
    )

    processed_rgb = cv2.cvtColor(
        processed_bgr,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(processed_rgb)

# =====================================================
# Gemmaレポート生成
# =====================================================

def ask_gemma(
    result: str,
    confidence: float,
) -> str:

    prompt = f"""
あなたは品質検査エンジニアです。

検査対象：ネジ

判定結果：{result}

信頼度：{confidence:.2f}

この検査結果について
品質検査レポートを日本語で作成してください。

3〜5文程度で簡潔に説明してください。
"""

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 150,
                },
            },
            timeout=120,
        )

        response.raise_for_status()

        return response.json().get(
            "response",
            "レポート生成失敗"
        )

    except Exception as e:

        return f"Gemmaエラー: {str(e)}"

# =====================================================
# Web画面
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# =====================================================
# Health Check
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# =====================================================
# AI検査API
# =====================================================

@app.post("/inspect")
async def inspect(
    file: UploadFile = File(...)
):

    contents = await file.read()

    raw_image = Image.open(
        io.BytesIO(contents)
    )

    processed_image = (
        normalize_environment_changes(
            raw_image
        )
    )

    if SAVE_DEBUG_IMAGE:

        debug_path = (
            DEBUG_DIR /
            f"debug_{file.filename}"
        )

        processed_image.save(
            debug_path
        )

    # YOLO推論

    results = model(
        processed_image
    )

    probs = results[0].probs

    class_id = probs.top1

    confidence = float(
        probs.top1conf
    )

    label = results[0].names[
        class_id
    ]

    japanese_label = LABEL_MAP.get(
    label,
    label
)

    # YOLO推論

    results = model(
        processed_image
    )

    probs = results[0].probs

    # ==================================
    # 全クラスの確率を表示（追加）
    # ==================================

    print("\n===== Prediction =====")

    for i, p in enumerate(probs.data.tolist()):

        print(
            f"{results[0].names[i]} : {round(p * 100, 2)}%"
        )

    print("======================\n")

    class_id = probs.top1

    confidence = float(
        probs.top1conf
    )

    label = results[0].names[
        class_id
    ]

# Gemmaレポート生成
    report = ask_gemma(
        japanese_label,
        confidence
    )

    if label == "good":
        result_text = "✓ 良品"
    else:
        result_text = f"⚠ 不良品（{japanese_label}）"

    return {
        "result": result_text,
        "result_jp": japanese_label,
        "confidence": round(
            confidence,
            4
        ),
        "report": report,
        "ai_model": "YOLOv8 Classification",
        "llm_model": "Gemma3:4b"
    }