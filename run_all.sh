#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════
#  run_all.sh — 6개 학습 잡을 한 번에 제출 (VESSL CLI)
#
#  사용: vessl 로그인 후
#     bash run_all.sh
#  그러면 6개 정책 잡이 동시에 생성된다. 모두 끝난 뒤:
#     vessl run create -f vessl_aggregate.yaml
# ══════════════════════════════════════════════════════════════════
set -e

POLICIES=("ES" "SF_9_EF_1" "SF_7_EF_3" "BL_SF_5_EF_5" "EF_7_SF_3" "EF_9_SF_1")
SEED="${SEED:-42}"

echo "▶ 6개 정책 학습 잡 제출 (SEED=${SEED})"
for P in "${POLICIES[@]}"; do
  echo "  → submit POLICY_NAME=${P}"
  vessl run create -f vessl_train.yaml \
    --env POLICY_NAME="${P}" \
    --env SEED="${SEED}"
done

echo ""
echo "✅ 6개 학습 잡 제출 완료."
echo "   모두 SUCCESS 된 뒤 아래로 통합 분석을 실행하세요:"
echo "     vessl run create -f vessl_aggregate.yaml"
