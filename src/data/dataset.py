"""
환자 단위 종단 다중모달 데이터셋.

핵심 설계:
  1) split_seed 로만 train/val/test 분할이 결정됨 (모델 초기화 시드와 독립).
  2) modality_dropout_seed 로만 "이번 실험에서 어떤 (환자,방문,모달리티)를
     추가로 결측 처리해 강건성을 테스트할지"가 결정됨.
     => 즉 동일 split에서 dropout 패턴만 바꾸는 ablation이 가능하고,
        반대로 동일 dropout 패턴에서 split만 바꾸는 ablation도 가능함.
  3) ADNI 실데이터 연결 지점은 `load_adni_csv_source()` 하나로 모아두었고,
     반환 스키마는 synthetic.py와 동일하게 맞춰 모델/트레이너가 데이터
     출처를 몰라도 되게 함.
"""
from typing import List, Dict, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import generate_synthetic_dataset, SyntheticConfig, AXES, MODALITIES

# 실제 ADNI CSV 컬럼명이 다르면 여기만 고치면 됩니다.
COLUMN_MAP = {
    "ptid": "PTID",
    "viscode": "VISCODE",
    "date": "EXAMDATE",
    "dx": "DX",
    "cdrsb": "CDRSB",
    "adas13": "ADAS13",
    "mmse": "MMSE",
    "apoe4": "APOE4",
}

# 모델이 기대하는 모달리티별 관측 차원. 실데이터 컬럼 구성이 다르면
# 이 값과 dataset이 만드는 텐서 차원이 반드시 일치해야 합니다.
MODALITY_DIMS = {
    "blood": 3,
    "csf": 2,
    "mri": None,   # n_rois 로 런타임에 채움
    "pet": None,   # n_rois
    "genetics": 1,
    "cognition": 2,
}


DX_MAP = {1.0: 0, 2.0: 1, 3.0: 2}  # ADNI DXSUM 표준 코드: 1=CN, 2=MCI, 3=Dementia(AD)


def _read_dxsum(path: str):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df = df[["RID", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].copy()
    df["dx"] = df["DIAGNOSIS"].map(DX_MAP)
    df = df.dropna(subset=["dx"])  # 10.0 등 비표준 코드, NaN 제거
    df["dx"] = df["dx"].astype(int)
    df = df.rename(columns={"EXAMDATE": "date"})
    return df[["RID", "VISCODE", "date", "dx"]]


def _read_cdr(path: str):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df = df[["RID", "VISCODE", "VISDATE", "CDRSB"]].copy()
    df = df.rename(columns={"VISDATE": "date"})
    return df.dropna(subset=["CDRSB"])


def _read_mmse(path: str):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    df = df[["RID", "VISCODE", "VISDATE", "MMSCORE"]].copy()
    df = df.rename(columns={"VISDATE": "date"})
    return df[df["MMSCORE"] >= 0]  # 음수 코드(미실시 등) 제외


def _read_adas(path: str):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    # TOTAL13(ADAS-Cog13) 우선, 없으면 TOTSCORE(ADAS-Cog11) 사용
    score_col = "TOTAL13" if "TOTAL13" in df.columns else "TOTSCORE"
    df = df[["RID", "VISCODE", "VISDATE", score_col]].copy()
    df = df.rename(columns={"VISDATE": "date", score_col: "ADAS"})
    return df.dropna(subset=["ADAS"])


def _load_demographics(csv_path: str):
    """
    scripts/extract_demographics.R 로 뽑은 demographics_baseline.csv를 읽어
    RID -> np.array([AGE, PTGENDER, PTEDUCAT, PTMARRY, APOE4]) 딕셔너리로 변환.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = ["AGE", "PTGENDER", "PTEDUCAT", "PTMARRY", "APOE4"]
    available = [c for c in cols if c in df.columns]
    if len(available) < len(cols):
        missing = set(cols) - set(available)
        print(f"[dataset] demographics_csv에 다음 컬럼이 없어 0으로 채웁니다: {missing}")

    out = {}
    for _, row in df.iterrows():
        rid = int(row["RID"])
        vec = np.zeros(5, dtype=np.float32)
        for i, c in enumerate(cols):
            if c in available and not pd.isna(row[c]):
                vec[i] = float(row[c])
        out[rid] = vec
    return out


def load_adni_csv_source(data_cfg: dict) -> List[Dict]:
    """
    실제 ADNI CDR/MMSE/ADAS/DXSUM CSV를 읽어 synthetic.py와 동일한 종단
    스키마로 변환합니다.

    현재 상태: MRI/PET(공간 구조)와 blood/CSF/genetics(다른 모달리티)는
    아직 파일이 없어 전부 mask=False(결측)로 채워 넣습니다. 즉 지금 이
    로더가 만드는 데이터로는 "축(K) 간 결합 + 시간 동역학"만 검증되고,
    "ROI 공간 위 확산"은 MRI/PET 확보 후에 검증됩니다 — 모델 코드는
    그대로 두고 데이터만 채워지면 자동으로 확장됩니다 (mask가 True로
    바뀌는 순간부터 해당 모달리티가 loss에 반영되는 구조이기 때문).
    """
    import pandas as pd
    import numpy as np

    adni_cfg = data_cfg["adni"]
    required = ["dxsum_csv", "cdr_csv", "mmse_csv", "adas_csv"]
    missing = [k for k in required if not adni_cfg.get(k)]
    if missing:
        raise ValueError(
            f"data.mode='adni_csv' 이지만 필수 경로가 비어 있습니다: {missing}. "
            f"configs/default.yaml 의 data.adni.* 를 채워주세요."
        )

    dx = _read_dxsum(adni_cfg["dxsum_csv"])
    cdr = _read_cdr(adni_cfg["cdr_csv"])
    mmse = _read_mmse(adni_cfg["mmse_csv"])
    adas = _read_adas(adni_cfg["adas_csv"])

    # RID+VISCODE 기준 outer merge (같은 방문이라도 검사별로 실시 여부가 다를 수 있음)
    merged = dx.merge(cdr[["RID", "VISCODE", "CDRSB"]], on=["RID", "VISCODE"], how="outer")
    merged = merged.merge(adas[["RID", "VISCODE", "ADAS"]], on=["RID", "VISCODE"], how="outer")
    merged = merged.merge(mmse[["RID", "VISCODE", "MMSCORE"]], on=["RID", "VISCODE"], how="outer")

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.dropna(subset=["date"])
    merged = merged.sort_values(["RID", "date"])

    n_rois = data_cfg["n_rois"]
    min_visits = 2
    patients = []

    demographics_path = adni_cfg.get("demographics_csv")
    demographics_lookup = _load_demographics(demographics_path) if demographics_path else {}
    if demographics_path:
        print(f"[dataset] demographics 로드됨: {len(demographics_lookup)}명 "
              f"(병합 시 RID 불일치분은 mask=False로 처리)")
    else:
        print("[dataset] demographics_csv 미지정 -> 전원 demographics_mask=False (ablation: without)")

    for rid, g in merged.groupby("RID"):
        g = g.sort_values("date").reset_index(drop=True)
        # 진단은 결측이면 직전 방문 값을 이월(carry-forward); 첫 방문부터 없으면 제외
        g["dx"] = g["dx"].ffill()
        g = g.dropna(subset=["dx"])
        if len(g) < min_visits:
            continue

        n_visits = len(g)
        t0 = g["date"].iloc[0]
        visit_months = ((g["date"] - t0).dt.days / 30.44).to_numpy(dtype=np.float32)
        viscode = g["VISCODE"].astype(str).tolist()  # 예: ['bl','m06','m12',...]

        cdrsb = g["CDRSB"].to_numpy(dtype=np.float32)
        adas13 = g["ADAS"].to_numpy(dtype=np.float32)
        cog_mask = (~np.isnan(cdrsb)) & (~np.isnan(adas13))
        # 디코더는 결측 없는 (B,T,2) 텐서를 기대하므로 NaN은 0으로 채우고 mask로 무시
        cognition = np.stack([np.nan_to_num(cdrsb), np.nan_to_num(adas13)], axis=1)

        rid_int = int(rid)
        if rid_int in demographics_lookup:
            demographics = demographics_lookup[rid_int]
            demographics_mask = True
        else:
            demographics = np.zeros(5, dtype=np.float32)
            demographics_mask = False

        patients.append({
            "ptid": f"RID-{int(rid):04d}",
            "visit_months": visit_months,
            "viscode": viscode,
            "observations": {
                "blood": np.zeros((n_visits, 3), dtype=np.float32),
                "csf": np.zeros((n_visits, 2), dtype=np.float32),
                "mri": np.zeros((n_visits, n_rois), dtype=np.float32),
                "pet": np.zeros((n_visits, n_rois), dtype=np.float32),
                "genetics": np.zeros((1,), dtype=np.float32),
                "cognition": cognition,
            },
            "mask": {
                "blood": np.zeros(n_visits, dtype=bool),
                "csf": np.zeros(n_visits, dtype=bool),
                "mri": np.zeros(n_visits, dtype=bool),
                "pet": np.zeros(n_visits, dtype=bool),
                "genetics": np.zeros(1, dtype=bool),
                "cognition": cog_mask,
            },
            "dx": g["dx"].to_numpy(dtype=np.int64),
            "demographics": demographics,
            "demographics_mask": demographics_mask,
        })

    if not patients:
        raise ValueError(
            "병합 후 사용 가능한 환자가 0명입니다. 방문 코드(VISCODE) 정렬이나 "
            "min_visits 조건을 확인해주세요."
        )
    return patients


def _train_val_test_split(ptids: List[str], train_frac: float, val_frac: float, split_seed: int):
    rng = np.random.default_rng(split_seed)
    idx = np.arange(len(ptids))
    rng.shuffle(idx)
    n = len(idx)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def _apply_extra_modality_dropout(patients: List[Dict], max_dropout: float, seed: int):
    """
    강건성 실험용: 각 (환자, 방문, 모달리티) 조합에 독립적으로
    Bernoulli(p) 결측을 추가한다. p ~ Uniform(0, max_dropout) 환자마다 다르게
    부여해 "결측 비율이 다양한 상황"까지 함께 커버한다.
    """
    if max_dropout <= 0:
        return patients
    rng = np.random.default_rng(seed)
    droppable = [m for m in MODALITIES if m not in ("genetics",)]  # 유전정보는 보통 1회성, 제외
    for p in patients:
        p_rate = rng.uniform(0, max_dropout)
        for m in droppable:
            n_visits = len(p["mask"][m])
            extra_drop = rng.random(n_visits) < p_rate
            p["mask"][m] = p["mask"][m] & (~extra_drop)
    return patients


class ADNILatentDataset(Dataset):
    def __init__(self, patients: List[Dict], n_rois: int):
        self.patients = patients
        self.n_rois = n_rois

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        return self.patients[idx]


def build_datasets(data_cfg: dict, split_seed: int, modality_dropout_seed: int
                    ) -> Tuple[ADNILatentDataset, ADNILatentDataset, ADNILatentDataset]:
    if data_cfg["mode"] == "synthetic":
        syn_cfg = SyntheticConfig(
            n_patients=data_cfg["synthetic"]["n_patients"],
            n_visits_min=data_cfg["synthetic"]["n_visits_min"],
            n_visits_max=data_cfg["synthetic"]["n_visits_max"],
            visit_gap_months_mean=data_cfg["synthetic"]["visit_gap_months_mean"],
            n_rois=data_cfg["n_rois"],
            # 데이터 "생성" 자체는 split과 무관하게 고정해야 동일 환자 pool 위에서
            # split만 바뀌는 형태가 됨. 별도 data seed를 두고 싶다면 config에 추가하세요.
            seed=0,
        )
        patients = generate_synthetic_dataset(syn_cfg)
    elif data_cfg["mode"] == "adni_csv":
        patients = load_adni_csv_source(data_cfg)
    else:
        raise ValueError(f"알 수 없는 data.mode: {data_cfg['mode']}")

    patients = _apply_extra_modality_dropout(
        patients, data_cfg["max_modality_dropout"], modality_dropout_seed
    )

    ptids = [p["ptid"] for p in patients]
    train_idx, val_idx, test_idx = _train_val_test_split(
        ptids, data_cfg["train_frac"], data_cfg["val_frac"], split_seed
    )

    def subset(indices):
        return ADNILatentDataset([patients[i] for i in indices], data_cfg["n_rois"])

    return subset(train_idx), subset(val_idx), subset(test_idx)


def collate_fn(batch: List[Dict]):
    """
    환자마다 방문 수(n_visits)가 달라 padding이 필요합니다.
    반환 텐서 shape: (B, T_max, ...), 그리고 valid_mask: (B, T_max) bool.
    """
    B = len(batch)
    T_max = max(len(p["visit_months"]) for p in batch)

    def pad_1d(arr, T):
        out = np.zeros(T, dtype=arr.dtype)
        out[: len(arr)] = arr
        return out

    def pad_2d(arr, T):
        out = np.zeros((T,) + arr.shape[1:], dtype=arr.dtype)
        out[: arr.shape[0]] = arr
        return out

    visit_months = np.stack([pad_1d(p["visit_months"], T_max) for p in batch])
    valid_mask = np.stack([pad_1d(np.ones(len(p["visit_months"]), dtype=bool), T_max) for p in batch])
    dx = np.stack([pad_1d(p["dx"], T_max) for p in batch])

    obs = {}
    obs_mask = {}
    for m in MODALITIES:
        if m == "genetics":
            obs[m] = np.stack([p["observations"][m] for p in batch])  # (B, d) 1회성
            obs_mask[m] = np.stack([np.array([p["mask"][m][0]]) for p in batch])
        else:
            obs[m] = np.stack([pad_2d(p["observations"][m], T_max) for p in batch])
            obs_mask[m] = np.stack([pad_1d(p["mask"][m], T_max) for p in batch])

    z_true = None
    if "z_true" in batch[0]:
        K, R = batch[0]["z_true"].shape[1], batch[0]["z_true"].shape[2]
        z_true = np.zeros((B, T_max, K, R), dtype=np.float32)
        for i, p in enumerate(batch):
            z_true[i, : p["z_true"].shape[0]] = p["z_true"]

    demographics = np.stack([p.get("demographics", np.zeros(5, dtype=np.float32)) for p in batch])
    demographics_mask = np.array(
        [1.0 if p.get("demographics_mask", False) else 0.0 for p in batch], dtype=np.float32
    )

    out = {
        "ptid": [p["ptid"] for p in batch],
        "viscode": [p.get("viscode", []) for p in batch],  # 텐서화하지 않고 그대로 전달 (검증/디버깅용)
        "visit_months": torch.from_numpy(visit_months).float(),
        "valid_mask": torch.from_numpy(valid_mask).bool(),
        "dx": torch.from_numpy(dx).long(),
        "observations": {k: torch.from_numpy(v).float() for k, v in obs.items()},
        "obs_mask": {k: torch.from_numpy(v).bool() for k, v in obs_mask.items()},
        "demographics": torch.from_numpy(demographics).float(),
        "demographics_mask": torch.from_numpy(demographics_mask).float(),
    }
    if z_true is not None:
        out["z_true"] = torch.from_numpy(z_true).float()
    return out
