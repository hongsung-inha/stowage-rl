# VESSL Batch Job — 4개 정책 병렬 실행 가이드

노트북으로 정식 학습을 돌리는 대신, 4개 정책(ES/SF/EF/BL)을 각각 별도 Job으로
병렬 제출합니다. 전체 시간이 약 1/4로 줄고, 한 Job이 실패해도 나머지는 영향받지 않습니다.

---

## 0. 먼저 — 노트북 학습 중단

지금 노트북에서 돌고 있는 정식 학습은 더 진행할 필요가 없습니다.
- 노트북 상단 ■ (Interrupt the kernel) 버튼으로 학습 중단
- Workspace 우상단 **Pause** 로 과금 중단 (Batch Job 은 별도 자원에서 돌기 때문에
  Workspace 를 켜둘 필요 없음)

---

## 1. 변경된 파일 (이번 업데이트)

노트북에서 겪은 두 문제를 Batch Job 에 미리 반영했습니다.

| 문제 | 반영 |
|------|------|
| `No module named 'scipy'` | `requirements.txt` 에 `scipy` 추가 |
| `FileNotFoundError: 'fc-cache'` | ① 실행 명령에 `apt-get install fontconfig fonts-nanum` 추가 ② 코드의 폰트 함수를 try/except 로 보호 |

**새 파일 목록**
- `config_ES.yaml`, `config_SF.yaml`, `config_EF.yaml`, `config_BL.yaml` — 정책별 설정 (각 1개 정책만 학습)
- `vessl_run_ES.yaml` ~ `vessl_run_BL.yaml` — 정책별 Job 설정
- `requirements.txt` (scipy 추가), `stowage_core.py` (폰트 함수 수정), `train_ppo.py` (변경 없음)

---

## 2. 코드 준비 — 둘 중 하나

Batch Job 은 깨끗한 컨테이너에서 시작하므로, 코드가 컨테이너에 들어가야 합니다.

### 방법 A — GitHub (권장)
1. 아래 파일들을 GitHub 저장소에 올림:
   `stowage_core.py`, `train_ppo.py`, `requirements.txt`,
   `config_ES.yaml`, `config_SF.yaml`, `config_EF.yaml`, `config_BL.yaml`
2. `vessl_run_*.yaml` 각 파일의 `import` 줄을 본인 저장소 주소로 교체:
   ```yaml
   import:
     /root/code: git://github.com/<YOUR_ID>/<YOUR_REPO>
   ```

### 방법 B — 코드 업로드 (저장소 없을 때)
1. `vessl_run_*.yaml` 에서 `import:` 블록 두 줄을 삭제
2. VESSL 콘솔에서 Job 생성 시 코드 파일을 직접 업로드

---

## 3. resources 수정 (필수)

`vessl_run_*.yaml` 4개 모두에서 본인 환경에 맞게 수정:
```yaml
resources:
  cluster: <본인 클러스터명>     # 노트북과 같은 클러스터 권장
  preset:  <본인 GPU preset>     # 예: A100:1 (노트북에서 A100 썼으므로 동일하게)
```
사용 가능한 preset 이름은 VESSL 콘솔 또는 `vessl cluster list-resources` 로 확인.

---

## 4. 병렬 제출

VESSL CLI 가 설치된 터미널(로컬 PC 또는 Workspace 터미널)에서:

```bash
vessl run create -f vessl_run_ES.yaml
vessl run create -f vessl_run_SF.yaml
vessl run create -f vessl_run_EF.yaml
vessl run create -f vessl_run_BL.yaml
```

네 줄을 연달아 실행하면 4개 Job 이 동시에 돕니다.
(콘솔 UI 에서 YAML 업로드로 제출해도 동일.)

> ⚠️ 4개 Job 이 각각 GPU 1개씩 = 동시에 GPU 4개를 점유합니다.
> 조직 GPU 쿼터가 부족하면 2개씩 나눠 제출하세요.

---

## 5. 진행 확인 / 결과 회수

- VESSL 콘솔의 Runs(또는 Jobs) 목록에서 4개 Job 의 Logs/Metrics 확인
- 각 Job 종료 후 `/output`(artifact)에 결과 보존:
  ```
  /output/results/
    ├── checkpoints/<exp>_<pol>_seed42.zip   # 학습된 PPO 모델
    ├── tb/<exp>/                            # TensorBoard
    ├── <exp>_<ts>/training_log.csv          # round·level·reward·KPI
    └── run_summary.json
  ```
- 콘솔에서 artifact 다운로드 → 4개 정책 결과를 합쳐 비교/그래프/통계 분석

---

## 6. 시드 늘리기 (SCIE 재현성, 선택)

논문용으로 다중시드가 필요하면 `config_*.yaml` 의 seeds 만 수정:
```yaml
seeds: [42, 123, 456, 2024]
```
한 Job 안에서 시드를 순차 처리합니다. 더 빠르게 하려면 (정책 × 시드) 조합마다
config 를 복제해 Job 수를 늘리면 됩니다.

---

## 참고 — 노트북 vs Batch Job

| | 노트북 | Batch Job |
|---|---|---|
| 브라우저 닫으면 | 중단 위험 | 계속 실행 |
| 결과 보존 | 수동 (`/shared` 복사) | 자동 (artifact) |
| 4정책 시간 | 순차 (≈4배) | 병렬 (≈1배) |
| 적합 단계 | 디버깅·소규모 | 정식 실험 |
