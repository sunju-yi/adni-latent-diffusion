# ADNI Latent Coupled Reaction–Diffusion (ADNI-LCRD)

Alzheimer's disease를 **다축(A/T/I/N/V) 잠재 병리 상태의 결합 반응-확산 동역학
추론 문제**로 정식화한 연구를 위한 코드베이스입니다.

Proposal의 Problem Statement (§1.5) 를 그대로 구현합니다:

```
dz(t) = [-L_phi(t) z(t) + r_theta(z(t))] dt + Sigma^{1/2}(z(t)) dW(t)
x_m(t) ~ P_psi_m( x_m | Pi_m z(t) )
```

## 왜 이렇게 짰는가 (설계 원칙)

1. **Colab → GPU 서버 이동을 전제로 설계**: 모든 학습 상태(모델, 옵티마이저,
   스케줄러, RNG 상태, epoch/step)를 매 checkpoint마다 저장하고, 재실행 시
   `latest.pt`를 자동 탐지해 이어서 학습합니다. Colab 세션이 끊겨도 데이터 손실이
   없습니다.
2. **재현성과 "데이터셋 조합"을 코드 레벨에서 분리**: `split_seed`(환자를
   train/val/test로 나누는 시드), `modality_dropout_seed`(어떤 모달리티를
   결측 처리해 강건성 실험을 할지 결정하는 시드), `model_init_seed`(가중치
   초기화 시드)를 **독립적으로** 지정할 수 있습니다. 이 세 값의 조합이 곧
   하나의 "실험 경우의 수"가 되고, 실험 ID는 이 조합의 해시로 자동 생성됩니다.
3. **실데이터가 없어도 파이프라인 검증 가능**: `data.mode: synthetic` 이면
   ADNI 스키마를 모사한 가짜 종단 다중모달 데이터를 생성해 모델/학습루프
   전체를 지금 바로 Colab에서 돌려볼 수 있습니다. ADNI 접근 시
   `data.mode: adni_csv` 로 전환하면 됩니다 (아래 스키마 참고).
4. **실험 추적은 Git 브랜치+태그 기반**: `scripts/new_experiment.sh` 가
   config 해시로 브랜치를 만들고, 학습이 끝나면 로그/체크포인트 메타데이터를
   커밋합니다. (주의: 이 샌드박스는 네트워크가 차단되어 있어 제가 직접
   `git push`를 실행해드릴 수 없습니다 — 아래 "GitHub 연동" 섹션에서
   사용자가 로컬/Colab에서 실행할 명령을 정리했습니다.)

## 디렉토리 구조

```
adni-latent-diffusion/
├── configs/default.yaml        # 모든 하이퍼파라미터 + 시드
├── src/
│   ├── utils/seed.py           # 시드 고정, RNG 유틸
│   ├── utils/checkpoint.py     # 저장/재개
│   ├── utils/logging_utils.py  # JSONL 학습 로그
│   ├── data/synthetic.py       # ADNI 스키마 모사 synthetic generator
│   ├── data/dataset.py         # 데이터셋 클래스 (실데이터/synthetic 공용)
│   ├── models/modules.py       # 인코더, reaction net, graph diffusion, emission
│   └── models/latent_dynamics.py # 전체 모델 (SDE 적분 + loss)
│   └── training/trainer.py     # 학습 루프 (resume 지원)
├── scripts/train.py            # 진입점
├── scripts/new_experiment.sh   # git 실험 브랜치 생성 스크립트
└── experiments/                # 실행 결과 (config별 하위 폴더, git-tracked)
```

## 지금 바로 실행 (Colab, synthetic 데이터)

```bash
pip install torch numpy pandas pyyaml networkx
python scripts/train.py --config configs/default.yaml
```

세션이 끊긴 뒤 동일 명령을 다시 실행하면 `experiments/<exp_id>/checkpoints/latest.pt`
를 찾아 자동으로 이어서 학습합니다.

## 다른 데이터 조합으로 실험 반복

```bash
python scripts/train.py --config configs/default.yaml \
    --set experiment.split_seed=1 experiment.modality_dropout_seed=7 \
    --set data.max_modality_dropout=0.3
```

`--set key=value` 로 config의 어떤 값도 CLI에서 덮어쓸 수 있습니다. 조합이
바뀔 때마다 `experiment.exp_id`가 자동으로 재계산되어 별도 폴더에 저장되므로
기존 실험을 덮어쓰지 않습니다.

---

## ⚠️ 사용자가 직접 결정해야 하는 부분

아래 항목들은 제가 임의로 정하면 연구 설계 자체가 흔들릴 수 있는 지점이라
default 값만 넣어두고 실제 결정은 남겨두었습니다.

1. **ROI 파킬레이션(atlas)**: 기본값은 Desikan-Killiany 68 ROI로 가정
   (`data.n_rois: 68`). Destrieux(148) 등 다른 atlas를 쓸지 결정 필요 —
   ADNI에서 이미 추출된 FreeSurfer ROI 테이블(UCSF 파생 테이블)의 atlas와
   일치시켜야 합니다.
2. **정확히 어떤 ADNI 테이블을 쓸지**: 아래 스키마는 일반적인 조합을
   가정한 것입니다. 실제로는 `ADNIMERGE`, `UCSFFSX(51/6)`(MRI ROI volume),
   PET SUVR 테이블(`UCBERKELEYAV45`/`UCBERKELEYAV1451` 등), CSF
   (`UPENNBIOMK`), APOE 유전형 테이블 중 **버전(예: UCSFFSX51 vs
   UCSFFSX6)**을 확정해야 컬럼명이 확정됩니다.
3. **방문 정렬 윈도우**: 서로 다른 모달리티가 같은 "시점"으로 취급될
   허용 오차 (`data.adni.visit_window_months`, 기본 6개월). 임상적으로
   타당한 값인지 검토 필요.
4. **약한 지도(anchor) 변수**: identifiability 제약에 쓸 인지 점수를
   CDR-SB로 할지 ADAS-Cog13으로 할지 (`loss.anchor_weight` 관련). 코드는
   CDR-SB를 기본값으로 가정합니다.
5. **진단/전환(conversion) 라벨 정의**: MCI→AD 전환을 "몇 개월 이내"로
   정의할지 (보통 24개월을 많이 쓰지만 연구 목적에 따라 다름).
6. **latent_dim_per_axis, hidden_dim 등 모델 용량**: Colab 무료 GPU(T4,
   ~15GB) 기준으로 안전한 기본값을 넣었습니다. 2주 후 확보하시는 GPU 사양에
   따라 늘릴 수 있습니다 — 사양을 알려주시면 맞춰 조정해드릴 수 있습니다.
7. **실험 추적 플랫폼**: 지금은 순수 Git + JSONL 로그로 구성했습니다.
   Weights & Biases 같은 툴을 병행할지 여부(협업/시각화 편의성 vs 추가
   의존성)를 결정해 주시면 `trainer.py`에 훅을 추가해드리겠습니다.

## ADNI CSV 스키마 (예상, `data.mode: adni_csv`일 때)

`configs/default.yaml`의 `data.adni.*` 경로에 아래 컬럼을 포함한 CSV를
지정하면 됩니다 (컬럼명은 실제 ADNI 다운로드본에 맞게
`src/data/dataset.py`의 `COLUMN_MAP`에서 한 번만 매핑해주면 됩니다):

- `adnimerge_csv`: `PTID, VISCODE, EXAMDATE, DX, CDRSB, ADAS13, MMSE, APOE4`
- `mri_roi_csv`: `PTID, VISCODE, ROI_1 ... ROI_68` (volume/thickness)
- `pet_roi_csv`: `PTID, VISCODE, TRACER, ROI_1_SUVR ... ROI_68_SUVR`
- `csf_csv`: `PTID, VISCODE, ABETA, TAU, PTAU`
- `apoe_csv`: `PTID, APOE_GENOTYPE`

## GitHub 연동 (사용자가 로컬/Colab에서 실행)

이 샌드박스는 네트워크가 차단되어 있어 제가 직접 push할 수 없습니다.
Colab 또는 로컬에서 아래와 같이 진행하시면 됩니다:

```bash
cd adni-latent-diffusion
git init
git add .
git commit -m "init: ADNI-LCRD scaffold"
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main

# 이후 새 실험 조합마다:
bash scripts/new_experiment.sh   # config 해시로 브랜치 생성 + 커밋
git push origin <생성된 브랜치명>
```

`experiments/` 폴더는 기본적으로 git에 포함되도록 했습니다(코드가 아니라
"연구 로그"이므로 추적 대상으로 삼는 것을 권장합니다). 체크포인트(`*.pt`)
파일만 `.gitignore`로 제외했습니다 — GitHub 파일 크기 제한 때문입니다.
체크포인트는 Google Drive나 Git LFS로 별도 관리하시는 것을 권장합니다.
