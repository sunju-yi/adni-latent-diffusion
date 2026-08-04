"""
build_datasets()가 만든 데이터가 실제로 "환자별 시간에 따른 상태 변화"를
제대로 담고 있는지 눈으로 확인하는 스크립트입니다. 모델 학습과 무관하게
데이터 파이프라인만 검증합니다.

사용법:
  python scripts/inspect_trajectories.py --config configs/default.yaml --n 5
  python scripts/inspect_trajectories.py --config configs/default.yaml --n 5 --plot
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config, apply_overrides
from src.data.dataset import build_datasets


def print_patient_trajectory(p: dict):
    print(f"\n--- {p['ptid']} ({len(p['visit_months'])} visits) ---")
    header = f"{'viscode':>8} {'month':>7} {'CDRSB':>7} {'ADAS13':>7} {'dx':>4} " \
             f"{'demo?':>6}"
    print(header)
    print("-" * len(header))
    cog = p["observations"]["cognition"]
    cog_mask = p["mask"]["cognition"]
    for t in range(len(p["visit_months"])):
        vc = p["viscode"][t] if "viscode" in p else "-"
        month = p["visit_months"][t]
        cdrsb = f"{cog[t,0]:.1f}" if cog_mask[t] else "  NA"
        adas = f"{cog[t,1]:.1f}" if cog_mask[t] else "  NA"
        dx = int(p["dx"][t])
        demo = "O" if p.get("demographics_mask", False) else "X"
        print(f"{vc:>8} {month:7.1f} {cdrsb:>7} {adas:>7} {dx:>4} {demo:>6}")


def summarize_monotonicity(patients):
    """
    CDR-SB가 방문이 진행될수록 대체로 비감소(=악화 또는 유지) 하는 비율을 계산.
    임상적으로 완전히 단조증가일 필요는 없지만(측정 노이즈, 일시적 호전 있음),
    전반적으로 우상향 추세여야 "시간에 따른 상태 변화"가 데이터에 실제로
    담겨 있다는 최소한의 sanity check가 됩니다.
    """
    import numpy as np
    trend_up = 0
    total = 0
    for p in patients:
        cog = p["observations"]["cognition"]
        mask = p["mask"]["cognition"]
        vals = cog[mask, 0]
        if len(vals) < 2:
            continue
        total += 1
        if vals[-1] >= vals[0]:
            trend_up += 1
    if total == 0:
        print("CDR-SB가 2회 이상 관측된 환자가 없습니다.")
        return
    print(f"\nCDR-SB baseline -> 마지막 관측 방문 사이 '악화 또는 유지' 비율: "
          f"{trend_up}/{total} ({trend_up/total:.1%})")
    print("(임상적으로 100%일 필요는 없지만, 대체로 50~60% 이상이면 시간축이 "
          "제대로 의미 있는 신호를 담고 있다는 뜻입니다.)")


def plot_sample(patients, out_path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    shown = 0
    for p in patients:
        cog = p["observations"]["cognition"]
        mask = p["mask"]["cognition"]
        months = p["visit_months"][mask]
        cdrsb = cog[mask, 0]
        if len(months) < 2:
            continue
        ax.plot(months, cdrsb, marker="o", alpha=0.7, label=p["ptid"])
        shown += 1
        if shown >= n:
            break
    ax.set_xlabel("Months since baseline")
    ax.set_ylabel("CDR-SB")
    ax.set_title(f"Sample patient CDR-SB trajectories (n={shown})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\n그래프 저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n", type=int, default=5, help="출력할 샘플 환자 수")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    train_ds, val_ds, test_ds = build_datasets(
        cfg["data"],
        split_seed=cfg["experiment"]["split_seed"],
        modality_dropout_seed=cfg["experiment"]["modality_dropout_seed"],
    )
    all_patients = train_ds.patients + val_ds.patients + test_ds.patients
    print(f"전체 환자 수: {len(all_patients)} (train={len(train_ds)}, "
          f"val={len(val_ds)}, test={len(test_ds)})")

    for p in all_patients[: args.n]:
        print_patient_trajectory(p)

    summarize_monotonicity(all_patients)

    if args.plot:
        os.makedirs("experiments/_inspect", exist_ok=True)
        plot_sample(all_patients, "experiments/_inspect/sample_trajectories.png", n=args.n)


if __name__ == "__main__":
    main()
