import sys
import os

# 프로젝트 루트 경로 등록
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from server.main import app  # noqa: F401  — Vercel이 `app` 을 ASGI 앱으로 인식
