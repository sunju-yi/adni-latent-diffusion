"""
ADNI 실데이터 접근 전에 전체 파이프라인(모델/학습/체크포인트/재개)을
검증하기 위한 synthetic longitudinal multimodal generator.

생성 원리: 환자별로 잠재 축 (A, T, I, N, V)이 실제 우리 모델과 동일한
reaction-diffusion 형태로 진짜 진행하도록 만들고, 거기서 noisy한
모달리티 관측치를 뽑아냅니다. 즉 "우리 모델이 정확히 이 생성과정을
복원할 수 있어야 한다"는 최소한의 sanity check 역할도 겸합니다.

실제 ADNI로 전환 시에는 이 파일 대신 dataset.py의 ADNIRealSource를 사용하며,
반환 스키마(딕셔너리 구조)는 동일하게 유지됩니다.
"""
from dataclasses import dataclass
from typing import List, Dict
import numpy as np


AXES = ["A", "T", "I", "N", "V"]
MODALITIES = ["blood", "csf", "mri", "pet", "genetics", "cognition"]
DEMOGRAPHICS_DIM = 5  # [AGE, PTGENDER, PTEDUCAT, PTMARRY, APOE4]


@dataclass
class SyntheticConfig:
    n_patients: int = 200
    n_visits_min: int = 3
    n_visits_max: int = 8
    visit_gap_months_mean: float = 6.0
    n_rois: int = 68
    seed: int = 0
    include_demographics: bool = True  # with/without ablation 파이프라인 테스트용


def _simulate_patient_trajectory(rng: np.random.Generator, n_rois: int, n_visits: int):
    """
    아주 단순화된 reaction-diffusion 시뮬레이터.
    z: (n_visits, K=5, n_rois)
    """
    K = len(AXES)
    z = np.zeros((n_visits, K, n_rois), dtype=np.float32)

    # 개인별 발병 성향 (genetic-like global scalar)
    genetic_risk = rng.normal(0, 1)

    # 초기 상태: A axis에 국소적 seed
    seed_roi = rng.integers(0, n_rois)
    z0 = np.zeros((K, n_rois), dtype=np.float32)
    z0[0, seed_roi] = 0.5 + 0.3 * max(genetic_risk, 0)
    z[0] = z0

    # 간단한 연결성: 인접 ROI index끼리 연결된 원형 그래프 (실제론 connectome)
    L = np.zeros((n_rois, n_rois), dtype=np.float32)
    for i in range(n_rois):
        j = (i + 1) % n_rois
        L[i, i] += 1
        L[j, j] += 1
        L[i, j] -= 1
        L[j, i] -= 1
    L /= n_rois

    dt = 1.0
    for t in range(1, n_visits):
        zt = z[t - 1].copy()
        # diffusion: 각 축이 그래프 위에서 확산
        diffusion = -np.einsum("rs,ks->kr", L, zt) * 0.3
        # reaction: A -> T -> I/N, N -> 인지저하 (cognition은 emission에서 다룸)
        A, T, I, N, V = zt
        rA = -0.05 * A
        rT = 0.15 * np.clip(A, 0, None) - 0.05 * T
        rI = 0.10 * np.clip(T, 0, None) - 0.08 * I
        rN = 0.08 * np.clip(T, 0, None) + 0.05 * np.clip(I, 0, None) - 0.03 * N
        rV = 0.02 * rng.normal(size=N.shape) - 0.02 * V
        reaction = np.stack([rA, rT, rI, rN, rV], axis=0)

        noise = rng.normal(0, 0.02, size=zt.shape).astype(np.float32)
        z[t] = np.clip(zt + dt * (diffusion + reaction) + noise, -3, 5)

    return z  # (n_visits, K, n_rois)


def generate_synthetic_dataset(cfg: SyntheticConfig) -> List[Dict]:
    """
    반환: 환자별 dict의 리스트. 각 dict는 dataset.py가 기대하는 공용 스키마.
    {
      "ptid": str,
      "visit_months": np.ndarray (n_visits,)  # baseline 대비 개월 수
      "z_true": np.ndarray (n_visits, K, R)   # 검증/디버깅용 (실데이터엔 없음)
      "observations": {
          "blood": (n_visits, d_blood) or None per-visit via mask,
          "csf": (n_visits, d_csf),
          "mri": (n_visits, R),
          "pet": (n_visits, R),
          "genetics": (d_gen,)  # baseline만 (한 번 측정)
          "cognition": (n_visits, 2)  # [CDRSB, ADAS13]
      },
      "mask": {modality: (n_visits,) bool},
      "dx": (n_visits,) int  # 0=CN, 1=MCI, 2=AD
    }
    """
    rng = np.random.default_rng(cfg.seed)
    dataset = []

    for p in range(cfg.n_patients):
        n_visits = int(rng.integers(cfg.n_visits_min, cfg.n_visits_max + 1))
        visit_months = np.cumsum(
            [0] + list(rng.normal(cfg.visit_gap_months_mean, 1.0, size=n_visits - 1))
        ).astype(np.float32)
        visit_months = np.clip(visit_months, 0, None)

        z_true = _simulate_patient_trajectory(rng, cfg.n_rois, n_visits)

        # emission: 각 모달리티는 z의 특정 부분공간에 대한 noisy projection
        A, T, I, N, V = z_true[:, 0], z_true[:, 1], z_true[:, 2], z_true[:, 3], z_true[:, 4]

        blood = (I.mean(axis=1, keepdims=True) + rng.normal(0, 0.1, (n_visits, 3))).astype(np.float32)
        csf = np.stack(
            [A.mean(axis=1) + rng.normal(0, 0.1, n_visits),
             T.mean(axis=1) + rng.normal(0, 0.1, n_visits)], axis=1
        ).astype(np.float32)
        mri = (-N + rng.normal(0, 0.15, N.shape)).astype(np.float32)  # 위축 proxy
        pet = (A + 0.5 * T + rng.normal(0, 0.15, A.shape)).astype(np.float32)
        genetics = np.array([rng.integers(0, 3)], dtype=np.float32)  # APOE4 allele count
        cognition = np.stack(
            [np.clip(N.mean(axis=1) * 6 + rng.normal(0, 0.3, n_visits), 0, 18),
             np.clip(N.mean(axis=1) * 20 + rng.normal(0, 1.0, n_visits), 0, 70)], axis=1
        ).astype(np.float32)

        cdrsb = cognition[:, 0]
        dx = np.digitize(cdrsb, bins=[0.5, 4.0]).astype(np.int64)  # 0/1/2

        mask = {
            "blood": np.ones(n_visits, dtype=bool),
            "csf": rng.random(n_visits) > 0.3,      # CSF는 흔히 결측
            "mri": np.ones(n_visits, dtype=bool),
            "pet": rng.random(n_visits) > 0.5,      # PET은 더 흔히 결측
            "genetics": np.ones(n_visits, dtype=bool),
            "cognition": np.ones(n_visits, dtype=bool),
        }

        if cfg.include_demographics:
            demographics = np.array([
                rng.normal(75, 8),                 # AGE
                float(rng.integers(0, 2)),         # PTGENDER
                rng.normal(15, 3),                 # PTEDUCAT
                float(rng.integers(0, 2)),         # PTMARRY
                float(rng.integers(0, 3)),         # APOE4
            ], dtype=np.float32)
            demographics_mask = True
        else:
            demographics = np.zeros(5, dtype=np.float32)
            demographics_mask = False

        dataset.append({
            "ptid": f"SYN-{p:04d}",
            "visit_months": visit_months,
            "viscode": [f"syn_v{t}" for t in range(n_visits)],
            "z_true": z_true,
            "observations": {
                "blood": blood, "csf": csf, "mri": mri, "pet": pet,
                "genetics": genetics, "cognition": cognition,
            },
            "mask": mask,
            "dx": dx,
            "demographics": demographics,
            "demographics_mask": demographics_mask,
        })

    return dataset
