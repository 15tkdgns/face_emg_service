# -*- coding: utf-8 -*-
"""
전체 ONNX 파일 일괄 테스트 (face_emg_2 신규 모델 포함)
"""
import os, sys, time
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE = r"c:\Users\15tkd\OneDrive\바탕 화면\Archive\프로젝트"

TARGETS = [
    # face_emg_2 신규
    ("face_emg_2 / efficientnet_v2_s",  r"face_emg_2\api\models\efficientnet_v2_s.onnx"),
    ("face_emg_2 / mobilenet_v2",        r"face_emg_2\api\models\mobilenet_v2.onnx"),
    ("face_emg_2 / face_detector",       r"face_emg_2\api\models\face_detector.onnx"),
    ("face_emg_2 / resnet18",            r"face_emg_2\api\models\resnet18.onnx"),
    # face_emg bible output
    ("bible A / densenet121_ce",         r"face_emg\output\bible\A_densenet121_ce\model.onnx"),
    ("bible B / densenet121_focal",      r"face_emg\output\bible\B_densenet121_focal\model.onnx"),
    ("bible C / efficientnet_ce",        r"face_emg\output\bible\C_efficientnet_ce\model.onnx"),
    ("bible D / densenet_focal_edge",    r"face_emg\output\bible\D_densenet121_focal_edge\model.onnx"),
    # 신희원 (재시도)
    ("신희원 / efficientnet_v2_s",       r"face_emg\신희원\best_efficientnet_v2_s.onnx"),
    ("face_emg_2 / 신희원",              r"face_emg_2\신희원\best_efficientnet_v2_s.onnx"),
    ("face_emg_2 / 한유승",              r"face_emg_2\한유승\emotion_model.onnx"),
]

def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def make_dummy(size=224):
    arr = np.random.randint(0,256,(size,size,3),dtype=np.uint8).astype(np.float32)/255.0
    norm = (arr - MEAN) / STD
    return norm.transpose(2,0,1)[np.newaxis].astype(np.float32)

def test(label, rel_path):
    path = os.path.join(BASE, rel_path)
    if not os.path.isfile(path):
        print(f"  [{label}]  파일없음")
        return None
    mb = os.path.getsize(path)/1024/1024
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1; opts.intra_op_num_threads = 1
    try:
        sess = ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"  [{label}]  로드실패: {str(e)[:60]}")
        return None

    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    inp_name = inp.name
    inp_shape = inp.shape

    # 입력 크기 결정
    if isinstance(inp_shape[2], int) and inp_shape[2] > 0:
        sizes = [inp_shape[2]]
    else:
        sizes = [224, 256, 384]

    for sz in sizes:
        dummy = make_dummy(sz)
        try:
            t0 = time.time()
            out_val = sess.run(None, {inp_name: dummy})[0]
            ms = (time.time()-t0)*1000
            # 출력 형태
            out_shape = out_val.shape
            if len(out_shape) == 2:
                logits = out_val[0]
            else:
                logits = out_val.flatten()
            probs = softmax(logits)
            n = len(probs)
            # 2회 더
            t_list = [ms]
            for _ in range(2):
                t0=time.time(); sess.run(None,{inp_name:dummy}); t_list.append((time.time()-t0)*1000)
            avg = sum(t_list)/len(t_list)
            print(f"  [{label}]  {mb:.1f}MB  입력={sz}  클래스={n}  avg={avg:.1f}ms  출력shape={out_shape}")
            return {'label':label,'mb':mb,'sz':sz,'n':n,'avg':avg}
        except Exception as e:
            continue

    print(f"  [{label}]  추론실패")
    return None

print("===== 전체 ONNX 탐색 =====\n")
results = [r for label, path in TARGETS if (r:=test(label,path))]

print(f"\n===== 요약 (성공 {len(results)}개) =====")
print(f"{'모델':<36} {'MB':>6} {'크기':>6} {'클래스':>6} {'속도ms':>8} {'배포가능'}")
print("-"*72)
for r in results:
    ok = "O" if r['mb'] < 50 else "X(대용량)"
    print(f"  {r['label']:<34} {r['mb']:>6.1f} {r['sz']:>6} {r['n']:>6} {r['avg']:>8.1f}  {ok}")
