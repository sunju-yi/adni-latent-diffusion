import os
import time
import torch
from torch.utils.data import DataLoader

from ..utils.checkpoint import CheckpointManager
from ..utils.logging_utils import JSONLLogger


class Trainer:
    def __init__(self, model, train_ds, val_ds, cfg, exp_dir, device):
        self.model = model.to(device)
        self.device = device
        self.cfg = cfg
        self.exp_dir = exp_dir

        from ..data.dataset import collate_fn
        g = torch.Generator()
        g.manual_seed(cfg["experiment"]["model_init_seed"])

        self.train_loader = DataLoader(
            train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
            collate_fn=collate_fn, num_workers=cfg["training"]["num_workers"],
            generator=g, drop_last=False,
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
            collate_fn=collate_fn, num_workers=cfg["training"]["num_workers"],
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"],
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10
        )

        self.ckpt_mgr = CheckpointManager(os.path.join(exp_dir, "checkpoints"))
        self.logger = JSONLLogger(os.path.join(exp_dir, "train_log.jsonl"))

        self.use_amp = cfg["training"]["amp"] and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.no_improve = 0

        self._maybe_resume()

    def _maybe_resume(self):
        if not self.cfg["training"]["resume"]:
            return
        latest = self.ckpt_mgr.latest_path()
        if latest is None:
            print("[trainer] 재개할 체크포인트 없음, 처음부터 시작합니다.")
            return
        step, epoch, extra = self.ckpt_mgr.load(
            latest, self.model, self.optimizer, self.scheduler, map_location=self.device
        )
        self.step = step
        self.epoch = epoch
        self.best_val_loss = extra.get("best_val_loss", float("inf"))
        self.no_improve = extra.get("no_improve", 0)
        print(f"[trainer] 체크포인트에서 재개: epoch={epoch}, step={step}, "
              f"best_val_loss={self.best_val_loss:.4f}")

    def _to_device(self, batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(self.device)
            elif isinstance(v, dict):
                out[k] = {kk: vv.to(self.device) for kk, vv in v.items()}
            else:
                out[k] = v
        return out

    def _run_epoch_train(self):
        self.model.train()
        for batch in self.train_loader:
            batch = self._to_device(batch)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                total, losses, _ = self.model(batch, self.cfg["loss"])

            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg["training"]["grad_clip"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.step += 1
            if self.step % self.cfg["training"]["log_every_steps"] == 0:
                self.logger.log(
                    split="train", step=self.step, epoch=self.epoch,
                    **{k: float(v.detach().cpu()) for k, v in losses.items()},
                )
            if self.step % self.cfg["training"]["checkpoint_every_steps"] == 0:
                self._save_checkpoint()

    @torch.no_grad()
    def _run_epoch_val(self):
        self.model.eval()
        totals = []
        for batch in self.val_loader:
            batch = self._to_device(batch)
            total, losses, _ = self.model(batch, self.cfg["loss"])
            totals.append(total.item())
        val_loss = sum(totals) / max(1, len(totals))
        self.logger.log(split="val", step=self.step, epoch=self.epoch, total=val_loss)
        return val_loss

    def _save_checkpoint(self):
        self.ckpt_mgr.save(
            self.step, self.epoch, self.model, self.optimizer, self.scheduler,
            extra={"best_val_loss": self.best_val_loss, "no_improve": self.no_improve},
        )

    def fit(self):
        max_epochs = self.cfg["training"]["epochs"]
        patience = self.cfg["training"]["early_stop_patience"]

        for epoch in range(self.epoch, max_epochs):
            self.epoch = epoch
            t0 = time.time()
            self._run_epoch_train()
            val_loss = self._run_epoch_val()
            self.scheduler.step(val_loss)
            dt = time.time() - t0
            print(f"[epoch {epoch}] val_loss={val_loss:.4f} ({dt:.1f}s) step={self.step}")

            improved = val_loss < self.best_val_loss - 1e-4
            if improved:
                self.best_val_loss = val_loss
                self.no_improve = 0
                self._save_checkpoint()
                best_path = os.path.join(self.exp_dir, "checkpoints", "best.pt")
                torch.save(self.model.state_dict(), best_path)
            else:
                self.no_improve += 1

            self._save_checkpoint()  # epoch 경계에서 항상 저장 (재개 안정성)

            if self.no_improve >= patience:
                print(f"[trainer] early stopping (patience={patience} 도달)")
                break

        self.logger.close()
