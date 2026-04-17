# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, '.')
from server.predictor import ModelManager, MODEL_REGISTRY
import numpy as np

print('=== MODEL_REGISTRY 확인 ===')
for mid, info in MODEL_REGISTRY.items():
    path = os.path.join('models', info['onnx'])
    exists = os.path.isfile(path)
    print(f'  [{mid}]  파일존재={exists}  클래스수={len(info["emotions"])}  감정={info["emotions"]}')

print()
print('=== han_yooseung 추론 테스트 ===')
mgr = ModelManager()
dummy = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
result = mgr.predict('han_yooseung', dummy)
if result:
    print(f'  감정={result["emotion"]}  신뢰도={result["confidence"]:.4f}  추론={result["infer_ms"]}ms')
    print(f'  scores 키 수={len(result["scores"])}  키={list(result["scores"].keys())}')
else:
    print('  [실패]')

print()
print('=== 전체 모델 available_models() ===')
for m in mgr.available_models():
    print(f'  id={m["id"]}  loaded={m["loaded"]}  file_exists={m["file_exists"]}')
