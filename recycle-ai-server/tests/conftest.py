# 2026-11-06: pytest가 recycle-ai-server 루트의 services/schemas 패키지를 import할 수 있게 경로를 추가함.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
