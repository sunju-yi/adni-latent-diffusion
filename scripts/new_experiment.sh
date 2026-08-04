#!/usr/bin/env bash
# 사용법: bash scripts/new_experiment.sh [config 경로] [-- 추가 --set 오버라이드들]
#
# 예:
#   bash scripts/new_experiment.sh configs/default.yaml
#   bash scripts/new_experiment.sh configs/default.yaml -- \
#       experiment.split_seed=1 data.max_modality_dropout=0.3
#
# 동작:
#   1) 주어진 오버라이드로 학습을 실행 (또는 이미 실행된 experiments/<exp_id>
#      를 그대로 커밋만 하고 싶다면 RUN_TRAIN=0으로 설정)
#   2) exp_id(설정 해시) 이름으로 git 브랜치를 만들고
#   3) config + 로그(체크포인트 제외, .gitignore로 무시됨)를 커밋
#
# 주의: 이 저장소가 아직 git init / remote 연결이 안 되어 있다면 먼저
#   git init && git remote add origin <repo_url>
# 을 실행해두세요. (README 'GitHub 연동' 섹션 참고)

set -euo pipefail

CONFIG=${1:-configs/default.yaml}
shift || true
if [[ "${1:-}" == "--" ]]; then shift; fi
OVERRIDES=("$@")

RUN_TRAIN=${RUN_TRAIN:-1}

SET_ARGS=()
for ov in "${OVERRIDES[@]}"; do
  SET_ARGS+=(--set "$ov")
done

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "[new_experiment] 학습 실행: python scripts/train.py --config $CONFIG ${SET_ARGS[*]:-}"
  python scripts/train.py --config "$CONFIG" "${SET_ARGS[@]:-}"
fi

# 방금 만들어진(또는 가장 최근) experiment 폴더 찾기
EXP_DIR=$(ls -td experiments/*/ | head -n 1)
EXP_ID=$(basename "$EXP_DIR")

echo "[new_experiment] exp_id = $EXP_ID"

if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "[new_experiment] 이 폴더는 아직 git 저장소가 아닙니다. 'git init'을 먼저 실행하세요."
  exit 1
fi

BRANCH="exp/${EXP_ID}"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
git add "$EXP_DIR/config.yaml" "$EXP_DIR/config.json" "$EXP_DIR/train_log.jsonl" 2>/dev/null || true
git commit -m "experiment: ${EXP_ID}" || echo "[new_experiment] 커밋할 변경사항 없음"

echo "[new_experiment] 완료. push하려면:"
echo "    git push -u origin $BRANCH"
