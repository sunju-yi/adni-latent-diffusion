"""
가볍고 의존성 없는 JSONL 로거.

W&B 등을 붙이고 싶다면 JSONLLogger.log()를 호출하는 지점에서
동일한 kwargs를 wandb.log(kwargs)로도 보내면 됩니다
(trainer.py의 self.logger.log(...) 호출부 참고).
"""
import json
import os
import time


class JSONLLogger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # append 모드: 재개(resume) 시 기존 로그 위에 이어 씀
        self._fh = open(path, "a")

    def log(self, **kwargs):
        kwargs["_ts"] = time.time()
        self._fh.write(json.dumps(kwargs) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
