"""
팀원 ONNX 모델 테스트 스크립트
- 입력/출력 텐서 형태 확인
- 더미 이미지로 추론 속도 측정
- 소프트맥스 출력 클래스 수 확인
"""
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGETS = {
    '한유승':   os.path.join(ROOT, '..', 'face_emg', '한유승',    'emotion_model.onnx'),
    '신희원':   os.path.join(ROOT, '..', 'face_emg', '신희원',    'best_efficientnet_v2_s.onnx'),
    'kang_mingoo': os.path.join(ROOT, '..', 'face_emg', 'kang_mingoo', 'resnet18_emotion.onnx'),
}

# 테스트용 이미지 크기 후보 (224, 256, 384, 480)
INPUT_SIZES = [224, 256, 300, 384]

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def make_dummy(size=224):
    img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    arr = img.astype(np.float32) / 255.0
    norm = (arr - MEAN) / STD
    return norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

def test_model(name, path):
    path = os.path.normpath(path)
    print(f"\n{'='*60}")
    print(f"  [{name}]  {os.path.basename(path)}")
    print(f"  경로: {path}")

    if not os.path.isfile(path):
        print("  [오류] 파일이 존재하지 않습니다.")
        return None

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  파일 크기: {size_mb:.1f} MB")

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1

    try:
        sess = ort.InferenceSession(path, sess_options=opts,
                                    providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"  [오류] 세션 로드 실패: {e}")
        return None

    # 입력/출력 정보
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    print(f"\n  [입력]  이름={inp.name}  shape={inp.shape}  type={inp.type}")
    print(f"  [출력]  이름={out.name}  shape={out.shape}  type={out.type}")

    # 동적 shape 판별
    inp_shape = inp.shape  # 예: [1, 3, None, None] 또는 [1, 3, 224, 224]
    if isinstance(inp_shape[2], int) and inp_shape[2] > 0:
        detected_size = inp_shape[2]
        print(f"  → 고정 입력 크기: {detected_size}x{detected_size}")
        sizes_to_try = [detected_size]
    else:
        print(f"  → 동적 입력 크기. {INPUT_SIZES} 순서로 시도...")
        sizes_to_try = INPUT_SIZES

    result = None
    for sz in sizes_to_try:
        dummy = make_dummy(sz)
        try:
            t0 = time.time()
            logits = sess.run(None, {inp.name: dummy})[0][0]
            elapsed = (time.time() - t0) * 1000
            probs   = softmax(logits)
            n_class = len(probs)
            pred    = int(probs.argmax())
            print(f"\n  [추론 성공] 입력크기={sz}  클래스수={n_class}  "
                  f"추론={elapsed:.1f}ms  최고신뢰={probs[pred]:.4f} (idx={pred})")

            # 3회 평균 추론 속도
            times = []
            for _ in range(3):
                t0 = time.time()
                sess.run(None, {inp.name: dummy})
                times.append((time.time() - t0) * 1000)
            avg_ms = sum(times) / len(times)
            print(f"  [속도]  3회 평균: {avg_ms:.1f}ms")

            result = {
                'name':     name,
                'path':     path,
                'size_mb':  size_mb,
                'input_sz': sz,
                'n_class':  n_class,
                'avg_ms':   avg_ms,
            }
            break
        except Exception as e:
            print(f"  [실패] 크기 {sz}: {e}")

    return result


if __name__ == '__main__':
    print("팀원 ONNX 모델 테스트 시작\n")
    results = []
    for name, path in TARGETS.items():
        r = test_model(name, path)
        if r:
            results.append(r)

    print(f"\n\n{'='*60}")
    print("  최종 요약")
    print(f"{'='*60}")
    print(f"{'이름':<14} {'크기(MB)':<10} {'입력px':<8} {'클래스':<8} {'속도(ms)':<10} {'배포적합'}")
    print("-" * 60)
    for r in results:
        suitable = "가능" if r['size_mb'] < 50 and r['avg_ms'] < 2000 else "주의"
        print(f"{r['name']:<14} {r['size_mb']:<10.1f} {r['input_sz']:<8} "
              f"{r['n_class']:<8} {r['avg_ms']:<10.1f} {suitable}")
