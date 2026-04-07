"""
ONNX Runtime 기반 감정인식 추론기.
PyTorch 없이 onnxruntime만으로 추론.
"""
import base64
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

EMOTIONS     = ['기쁨', '당황', '분노', '상처']
EMOTION_EMOJI = {'기쁨': '😄', '당황': '😳', '분노': '😡', '상처': '😢'}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
)

MODEL_REGISTRY = {
    'densenet121': {
        'label':       'DenseNet121',
        'description': '기본 전처리 · Best 모델',
        'onnx':        'densenet121.onnx',
        'color':       '#4F86C6',
        'val_acc':     0.8762,
        'f1_per':      {'기쁨': 0.968, '당황': 0.902, '분노': 0.860, '상처': 0.828},
        'use_clahe':   False,
        'use_edge':    False,
        'in_channels': 3,
    },
    'densenet121_clahe_edge': {
        'label':       'DenseNet121 + CLAHE + Edge',
        'description': 'CLAHE 평활화 + Canny 엣지 채널',
        'onnx':        'densenet121_clahe_edge.onnx',
        'color':       '#57B894',
        'val_acc':     0.8476,
        'f1_per':      {'기쁨': 0.959, '당황': 0.881, '분노': 0.813, '상처': 0.807},
        'use_clahe':   True,
        'use_edge':    True,
        'in_channels': 4,
    },
    'efficientnet_b0': {
        'label':       'EfficientNet-B0',
        'description': '기본 전처리',
        'onnx':        'efficientnet_b0.onnx',
        'color':       '#F4845F',
        'val_acc':     0.8262,
        'f1_per':      {'기쁨': 0.968, '당황': 0.846, '분노': 0.810, '상처': 0.760},
        'use_clahe':   False,
        'use_edge':    False,
        'in_channels': 3,
    },
    'efficientnet_b0_clahe_edge': {
        'label':       'EfficientNet-B0 + CLAHE + Edge',
        'description': 'CLAHE 평활화 + Canny 엣지 채널',
        'onnx':        'efficientnet_b0_clahe_edge.onnx',
        'color':       '#9b59b6',
        'val_acc':     0.8167,
        'f1_per':      {'기쁨': 0.968, '당황': 0.864, '분노': 0.798, '상처': 0.729},
        'use_clahe':   True,
        'use_edge':    True,
        'in_channels': 4,
    },
}

PIPELINE_IMAGES = {
    'comparison': 'static/comparison.png',
}


def _apply_clahe(img_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _extract_edge(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def detect_and_crop(img_bgr: np.ndarray):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
    )
    if len(faces) == 0:
        h, w = img_bgr.shape[:2]
        s = min(h, w)
        x1, y1 = (w - s) // 2, (h - s) // 2
        face_bgr = img_bgr[y1:y1 + s, x1:x1 + s]
        bbox = None
    else:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        pad_x = int(fw * 0.1)
        pad_y = int(fh * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_bgr.shape[1], x + fw + pad_x)
        y2 = min(img_bgr.shape[0], y + fh + pad_y)
        face_bgr = img_bgr[y1:y2, x1:x2]
        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]

    _, buf = cv2.imencode('.jpg', face_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    face_b64 = base64.b64encode(buf).decode('utf-8')
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return bbox, face_rgb, face_b64


def _preprocess(face_rgb: np.ndarray, use_clahe: bool, use_edge: bool) -> np.ndarray:
    face = cv2.resize(face_rgb, (224, 224))
    if use_clahe:
        face = _apply_clahe(face)
    face_f    = face.astype(np.float32) / 255.0
    face_norm = (face_f - MEAN) / STD
    rgb = face_norm.transpose(2, 0, 1)          # (3, H, W)
    if use_edge:
        edge = _extract_edge(face).astype(np.float32) / 255.0
        tensor = np.concatenate([rgb, edge[np.newaxis]], axis=0)  # (4, H, W)
    else:
        tensor = rgb
    return tensor[np.newaxis].astype(np.float32)  # (1, C, H, W)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class ModelManager:
    def __init__(self):
        self.sessions: dict[str, ort.InferenceSession] = {}

    def load_all(self):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1

        for mid, info in MODEL_REGISTRY.items():
            path = os.path.join(MODELS_DIR, info['onnx'])
            if not os.path.isfile(path):
                logger.warning(f'[{mid}] ONNX 없음: {path}')
                continue
            try:
                self.sessions[mid] = ort.InferenceSession(
                    path, sess_options=opts,
                    providers=['CPUExecutionProvider'],
                )
                logger.info(f'[{mid}] 로드 완료')
            except Exception as e:
                logger.error(f'[{mid}] 로드 실패: {e}')
        logger.info(f'로드된 모델: {list(self.sessions.keys())}')

    def available_models(self) -> list:
        return [
            {
                'id':          mid,
                'label':       info['label'],
                'description': info['description'],
                'color':       info['color'],
                'loaded':      mid in self.sessions,
                'val_acc':     info['val_acc'],
                'f1_per':      info['f1_per'],
            }
            for mid, info in MODEL_REGISTRY.items()
        ]

    def predict(self, model_id: str, face_rgb: np.ndarray) -> dict | None:
        if model_id not in self.sessions:
            return None
        info    = MODEL_REGISTRY[model_id]
        tensor  = _preprocess(face_rgb, info['use_clahe'], info['use_edge'])
        t0      = time.time()
        logits  = self.sessions[model_id].run(None, {'input': tensor})[0][0]
        elapsed = (time.time() - t0) * 1000
        probs   = _softmax(logits)
        pred    = int(probs.argmax())
        return {
            'emotion':    EMOTIONS[pred],
            'emoji':      EMOTION_EMOJI[EMOTIONS[pred]],
            'confidence': float(probs[pred]),
            'scores':     {e: float(probs[i]) for i, e in enumerate(EMOTIONS)},
            'infer_ms':   round(elapsed, 1),
        }

    def predict_all(self, face_rgb: np.ndarray) -> list:
        results = []
        for mid in MODEL_REGISTRY:
            if mid not in self.sessions:
                continue
            res = self.predict(mid, face_rgb)
            if res:
                res['model_id']    = mid
                res['model_label'] = MODEL_REGISTRY[mid]['label']
                res['color']       = MODEL_REGISTRY[mid]['color']
                results.append(res)
        return results
