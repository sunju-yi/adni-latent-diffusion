"""
재현성을 위한 시드 유틸.

이 프로젝트는 시드를 하나로 뭉치지 않고 목적별로 분리합니다:
  - model_init_seed        : 가중치 초기화, dropout 등 모델 내부 확률성
  - split_seed              : 환자를 train/val/test로 나누는 데 사용
  - modality_dropout_seed   : 어떤 (환자, 방문, 모달리티)를 결측 처리할지

이렇게 분리해두면 "같은 데이터 split인데 모델 초기화만 다르게" 같은
ablation을 독립적으로 돌릴 수 있습니다.
"""
import os
import random
import hashlib
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def torch_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def config_hash(cfg: dict, length: int = 8) -> str:
    """
    실험 설정(dict)을 결정론적으로 해시하여 exp_id에 사용.
    시드 조합이 하나라도 다르면 다른 해시가 나오므로,
    실험 폴더가 자동으로 분리됩니다.
    """
    payload = repr(sorted(cfg.items())).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]
