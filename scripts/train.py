"""
사용 예:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml \
      --set experiment.split_seed=1 --set data.max_modality_dropout=0.3

exp_id는 (실험 이름 + 세 종류의 시드 + 데이터/모델 핵심 설정)의 해시로
자동 생성되며, experiments/<exp_id>/ 아래에 config, 체크포인트, 로그가
저장됩니다. 동일 조합으로 다시 실행하면 자동으로 이어서 학습합니다
(training.resume: true 인 경우).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.utils.config import load_config, apply_overrides
from src.utils.seed import seed_everything, config_hash
from src.data.dataset import build_datasets
from src.models.latent_dynamics import LatentCoupledReactionDiffusion
from src.training.trainer import Trainer


def build_exp_id(cfg: dict) -> str:
    exp = cfg["experiment"]
    key_payload = {
        "name": exp["name"],
        "model_init_seed": exp["model_init_seed"],
        "split_seed": exp["split_seed"],
        "modality_dropout_seed": exp["modality_dropout_seed"],
        "data_mode": cfg["data"]["mode"],
        "max_modality_dropout": cfg["data"]["max_modality_dropout"],
        "n_rois": cfg["data"]["n_rois"],          # shape에 영향 -> 다르면 반드시 별도 실험
        "axes": cfg["data"]["axes"],               # shape에 영향
        "use_demographics": bool(cfg["data"]["adni"].get("demographics_csv")),
        "model": cfg["model"],
        "loss": cfg["loss"],
    }
    h = config_hash(key_payload)
    return f"{exp['name']}__{h}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    exp_id = build_exp_id(cfg)
    exp_dir = os.path.join(cfg["experiment"]["output_root"], exp_id)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[main] experiment dir: {exp_dir}")

    with open(os.path.join(exp_dir, "config.yaml"), "w") as f:
        import yaml
        yaml.safe_dump(cfg, f, allow_unicode=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    # 모델 초기화/학습 확률성은 model_init_seed로 고정
    seed_everything(cfg["experiment"]["model_init_seed"])

    train_ds, val_ds, test_ds = build_datasets(
        cfg["data"],
        split_seed=cfg["experiment"]["split_seed"],
        modality_dropout_seed=cfg["experiment"]["modality_dropout_seed"],
    )
    print(f"[main] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} 환자")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] device={device}")

    model = LatentCoupledReactionDiffusion(
        model_cfg=cfg["model"], n_rois=cfg["data"]["n_rois"], n_axes=len(cfg["data"]["axes"]),
    )

    trainer = Trainer(model, train_ds, val_ds, cfg, exp_dir, device)
    trainer.fit()

    # test split은 별도 스크립트(scripts/evaluate.py, 필요 시 추가 요청 주세요)에서
    # best.pt를 로드해 평가하는 것을 권장합니다 (train.py는 학습 전용으로 유지).
    print(f"[main] 학습 종료. best checkpoint: {exp_dir}/checkpoints/best.pt")


if __name__ == "__main__":
    main()
