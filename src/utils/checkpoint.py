"""
학습 상태 저장/재개.

Colab처럼 세션이 예고 없이 끊기는 환경을 전제로, 매 checkpoint마다
다음을 전부 저장합니다:
  - 모델 가중치 / optimizer / scheduler 상태
  - 현재 step, epoch
  - torch / cuda RNG 상태 (재개 시 확률적 흐름까지 최대한 동일하게)
  - extra (best_val_loss, no_improve_count 등 학습 루프가 필요로 하는 값)

latest.pt 는 매번 덮어써서 "가장 최근 지점"을 항상 가리키게 하고,
step_00000200.pt 같은 파일은 별도로 남겨 특정 시점으로 롤백할 수 있게 합니다.
"""
import os
import glob
import torch


class CheckpointManager:
    def __init__(self, ckpt_dir: str, keep_last_n: int = 5):
        self.ckpt_dir = ckpt_dir
        self.keep_last_n = keep_last_n
        os.makedirs(ckpt_dir, exist_ok=True)

    def latest_path(self):
        p = os.path.join(self.ckpt_dir, "latest.pt")
        return p if os.path.exists(p) else None

    def save(self, step, epoch, model, optimizer, scheduler=None, extra=None):
        payload = {
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "extra": extra or {},
        }
        step_path = os.path.join(self.ckpt_dir, f"step_{step:08d}.pt")
        torch.save(payload, step_path)
        torch.save(payload, os.path.join(self.ckpt_dir, "latest.pt"))
        self._prune()
        return step_path

    def _prune(self):
        ckpts = sorted(
            glob.glob(os.path.join(self.ckpt_dir, "step_*.pt")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
        for old in ckpts[: max(0, len(ckpts) - self.keep_last_n)]:
            try:
                os.remove(old)
            except OSError:
                pass

    def load(self, path, model, optimizer=None, scheduler=None, map_location="cpu"):
        payload = torch.load(path, map_location=map_location)
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        rng = payload.get("rng_state")
        if rng is not None:
            torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        return payload["step"], payload["epoch"], payload.get("extra", {})
