"""
Vercel 서버리스 진입점.
FastAPI ASGI 앱을 그대로 노출.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.main import app  # noqa: F401
