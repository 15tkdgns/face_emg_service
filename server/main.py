"""
감정인식 FastAPI 서비스 (ONNX Runtime 전용)

엔드포인트:
  GET  /api/health
  GET  /api/models
  POST /api/analyze
  POST /api/analyze/compare
  POST /api/analyze/base64
"""
import base64
import io
import logging
import os
import sys
import traceback as tb

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server.predictor import ModelManager, detect_and_crop

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title='Face Emotion API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

manager = ModelManager()  # lazy loading — 첫 요청 시 모델 자동 로드


def _validate_image(contents: bytes) -> bytes:
    """이미지 바이트 검증 및 최대 1280px 리사이즈."""
    try:
        img = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='이미지 디코딩 실패')
    w, h = img.size
    if max(w, h) > 1280:
        scale = 1280 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=90)
        return buf.getvalue()
    return contents


@app.get('/api/health')
def health():
    return {
        'status': 'ok',
        'models': list(manager.available_models()),
    }


@app.get('/api/models')
def get_models():
    return {'models': manager.available_models()}


@app.post('/api/analyze')
async def analyze(
    file:     UploadFile = File(...),
    model_id: str        = Form(default='densenet121'),
):
    try:
        contents = _validate_image(await file.read())
        bbox, face_rgb, face_b64 = detect_and_crop(contents)
        result = manager.predict(model_id, face_rgb)
        if result is None:
            raise HTTPException(status_code=503, detail=f'추론 실패: {model_id}')
        return {**result, 'face_b64': face_b64, 'face_detected': bool(bbox),
                'bbox': bbox, 'model_id': model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'analyze error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/analyze/compare')
async def analyze_compare(file: UploadFile = File(...)):
    try:
        contents = _validate_image(await file.read())
        bbox, face_rgb, face_b64 = detect_and_crop(contents)
        results = manager.predict_all(face_rgb)
        if not results:
            raise HTTPException(status_code=503, detail='로드된 모델 없음')
        return {'results': results, 'face_b64': face_b64,
                'face_detected': bool(bbox), 'bbox': bbox}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'compare error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/api/analyze/base64')
async def analyze_base64(payload: dict):
    try:
        image_b64 = payload.get('image_b64', '')
        model_id  = payload.get('model_id', 'densenet121')
        compare   = payload.get('compare', False)
        if not image_b64:
            raise HTTPException(status_code=400, detail='image_b64 없음')
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        contents = _validate_image(base64.b64decode(image_b64))
        bbox, face_rgb, face_b64 = detect_and_crop(contents)
        if compare:
            results = manager.predict_all(face_rgb)
            return {'results': results, 'face_b64': face_b64,
                    'face_detected': bool(bbox)}
        result = manager.predict(model_id, face_rgb)
        if result is None:
            raise HTTPException(status_code=503, detail=f'추론 실패: {model_id}')
        return {**result, 'face_b64': face_b64, 'face_detected': bool(bbox),
                'model_id': model_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'base64 error:\n{tb.format_exc()}')
        raise HTTPException(status_code=500, detail=str(e))
