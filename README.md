# VESSL 병렬 실행 — 안정:효율 비중 6-정책 (v17)

원본 노트북 `single_bay_6pod_ppo_v17_6policy_weighted.ipynb` 의 6개 정책
(ES / SF_9_EF_1 / SF_7_EF_3 / BL_SF_5_EF_5 / EF_7_SF_3 / EF_9_SF_1) 을
**VESSL 배치 잡으로 병렬 학습**하고, 결과를 **통합 분석**하는 패키지입니다.

VESSL은 한 잡 = GPU 1장이므로, "정책당 1개 잡 × 6개"를 동시에 띄워 물리적
병렬을 얻습니다. 6개가 끝나면 분석 잡 1개가 결과를 모아 Figure 1~6,
가설검증, Excel, RDB/LPG/RAG/SFT 데이터셋을 한 번에 생성합니다.

---

## 구조

```
stowage_core.py          # 노트북 Cell 2(정의)+Cell 3(CONFIG/POLICIES) 모듈화
                         #   - 환경변수 OUTPUT_ROOT 로 저장 루트 지정
train_policy.py          # [병렬 워커] POLICY_NAME 1개만 학습 → /output/<policy>/
aggregate.py             # [통합 잡] 6개 결과 로드 → 분석 셀 14개 실행
analysis_cells/          # 노트북 Cell 5~16 을 셀 단위 .py 로 분리 (코드 동일)
  05_fig1_cumulative_reward.py   ... 16_sft_dataset.py
vessl_train.yaml         # 학습 잡 정의 (6번 제출, POLICY_NAME 만 변경)
vessl_aggregate.yaml     # 통합 분석 잡 정의 (1번 실행)
run_all.sh               # 6개 학습 잡 일괄 제출 스크립트
requirements.txt
```

## 실행 흐름

### 0. 사전 준비 (1회)
- GitHub repo `hongsung-inha/stowage-rl` 에 이 폴더 내용을 push (public).
- VESSL object storage 볼륨 생성 (예: `stowage-output`).
- `vessl_train.yaml` / `vessl_aggregate.yaml` 의 `cluster`, `preset`,
  `volume`, repo `url` 을 본인 환경에 맞게 수정.

### 1. 6개 정책 병렬 학습
```bash
bash run_all.sh
# 또는 수동으로 6번:
vessl run create -f vessl_train.yaml --env POLICY_NAME=ES
vessl run create -f vessl_train.yaml --env POLICY_NAME=SF_9_EF_1
vessl run create -f vessl_train.yaml --env POLICY_NAME=SF_7_EF_3
vessl run create -f vessl_train.yaml --env POLICY_NAME=BL_SF_5_EF_5
vessl run create -f vessl_train.yaml --env POLICY_NAME=EF_7_SF_3
vessl run create -f vessl_train.yaml --env POLICY_NAME=EF_9_SF_1
```
각 잡은 `/output/<POLICY_NAME>/` 에 다음을 저장합니다:
- `model.zip` (SB3 PPO 모델), `result.pkl` (분석용 직렬화 결과),
  `meta.json`, `logger_artifacts/`, `_SUCCESS`.

### 2. 통합 분석 (6개 잡 완료 후)
```bash
vessl run create -f vessl_aggregate.yaml
```
`/output/_aggregate/` 에 산출물이 생성됩니다:
- `figures/.../fig1~fig6*.png` (학습곡선, KPI, 레이더, bay plan)
- 종합 비교표·가설검증 콘솔 로그
- `*_results.xlsx`, `appendix_reward_balance_v16.xlsx`,
  `*_BayPlan_Distributions_*.xlsx`, `*_RDB_*.xlsx`
- `*_RDB_LPG_*.zip`, `*_RAG_*.zip`, `rag_chunks.jsonl`,
  `*_sft_*.jsonl`, `xai_grounding.json`, 모델 번들 zip

## 가설검증 정책 매핑
통합 분석의 H1~H5 가설은 다음 대표 정책으로 평가합니다:
- 안정 대표 `STAB` = **SF_9_EF_1** (안정 Σ|w|=13.5)
- 효율 대표 `EFF` = **EF_9_SF_1** (효율 Σ|w|=13.5)
- 균형 기준 `BAL` = **BL_SF_5_EF_5** (안정:효율 = 7.5:7.5)
6정책 전체는 KPI 종합표·그림에 모두 표시됩니다.

## 단일 노드 폴백 (VESSL 없이 디버그)
한 머신에서 6개를 순차로 돌리려면:
```bash
OUTPUT_ROOT=./output POLICY_NAME=ALL python train_policy.py
OUTPUT_ROOT=./output python aggregate.py
```

## 주의 / VESSL 워크플로우 팁
- 6개 학습 잡과 분석 잡은 **반드시 같은 볼륨**(`stowage-output`)을
  `/output` 에 마운트해야 결과 공유가 됩니다.
- 결과 영속화: 볼륨을 마운트하지 않으면 잡 종료 시 결과가 사라집니다.
- repo 는 batch job 의 `git clone` 을 위해 **public** 이어야 합니다.
- 한글 폰트 깨짐 방지를 위해 command 에 `fonts-nanum`·`fontconfig`
  설치를 포함했습니다 (이미지에 따라 불필요할 수 있어 `|| true`).
- 분석 잡은 GPU 불필요 → `cpu-medium` 으로 비용 절감.
- `REQUIRE_ALL=0` 으로 제출하면 일부 정책만으로도 부분 분석이 가능합니다.

## 검증 상태
6개 정책 축소예산 학습 → 통합 분석을 로컬에서 실행해 **분석 셀 14/14 성공**,
Figure 1~6 + Excel + 데이터셋 생성까지 end-to-end 확인했습니다.
실제 실행은 `episodes_per_level` 가 풀 예산(Lv4 48K eps 등)이라 정책당
수 시간이 소요됩니다.
