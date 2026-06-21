# ══════════════════════════════════════════════════════════════════
#  ★ NEW (v14): SLM 파인튜닝 — instruction 학습 데이터 export  (슬라이드 ④)
#   · RL 결과를 (질문 → 근거 있는 답변) 쌍(SFT)으로 변환 — 가장 가치 있는 자산
#       (a) SFT JSONL : {instruction, input(컨테이너 제원), output(배치권고+사유)}
#       (b) reward 분해(R1~R15) → "왜 이 배치가 좋은가" 라벨 자동 생성
#       (c) 정책 대비(BL vs SF vs EF) → "안전 우선 vs 효율 우선" 설명 패턴
#       (d) text2Cypher 질문-쿼리 쌍 (NL2SQL/Cypher 생성 품질 향상)
#   · ★ REP_CACHE/SEQ_CACHE(Cell 14) + RAG(Cell 15) 재사용
# ══════════════════════════════════════════════════════════════════
import os, json
assert "REP_CACHE" in dir(), "REP_CACHE 없음 — Cell 14를 먼저 실행하세요."
REP = REP_CACHE; SEQ = SEQ_CACHE

SLM_DIR = "/content/results/slm"
os.makedirs(SLM_DIR, exist_ok=True)

sft = []   # SFT 레코드 리스트

# ── (a)+(b) 컨테이너 배치 권고 SFT (reward 분해 → 사유 라벨) ──
R15_LABELS = {
    "R6_cog": "heavy-down(무게중심 하강)", "R10_tier_match": "동일 POD 수평 그룹핑",
    "R4_order": "POD 반출순서 정렬", "R5_weight_bal": "행간 무게 균형",
    "R3_overstow": "재취급 회피", "R8_pod_band": "POD 밴드 품질",
}
for (pol, rnd_idx), d in REP.items():
    bp = d["bay_plan"]; m = d["metrics"]; rc = m.reward_components
    lv = SEQ[rnd_idx]["level"]
    # 사유 라벨: 양의 기여 상위 항목
    top = sorted([(k, v) for k, v in rc.items() if v > 0 and k in R15_LABELS],
                 key=lambda x: -x[1])[:2]
    reasons = ", ".join(R15_LABELS[k] for k, _ in top) or "운영 제약 충족"
    # 대표 컨테이너(첫 적재) 제원으로 instruction 예시 구성
    pods, wts = SEQ[rnd_idx]["pod"], SEQ[rnd_idx]["wt"]
    if len(pods) == 0:
        continue
    c_pod, c_wt = int(pods[0]), float(wts[0])
    # 첫 컨테이너의 실제 배치 위치(SF 기준 bay_plan에서 탐색)
    pod_grid, stack_h = bp["pod_grid"], bp["stack_h"]
    placed = None
    for rr in range(bp["n_rows"]):
        for t in range(int(stack_h[rr])):
            if int(pod_grid[rr, t]) == c_pod:
                placed = (rr, t); break
        if placed: break
    tier_word = "하부" if (placed and placed[1] <= bp["n_tiers"]//2) else "상부"
    sft.append({
        "instruction": "다음 컨테이너의 적재 위치를 권고하고 그 사유를 설명하라.",
        "input": f"40HC, {c_wt:.1f}t, POD={POD_NAMES[c_pod]}({c_pod}), "
                 f"bay={bp['n_rows']}R×{bp['n_tiers']}T, policy={pol}",
        "output": f"{tier_word} Tier 권장. 사유: {reasons}. "
                  f"(이 라운드 OSR={m.osr:.3f}, WBI={m.wbi:.3f}, PSR={m.psr:.3f})",
        "meta": {"policy": pol, "round_id": rnd_idx+1, "level": lv, "type": "placement"},
    })

# ── (c) 정책 대비 설명 패턴 (대표 3정책 BAL/STAB/EFF, 동일 라운드) ──
for rnd_idx in range(CONFIG["total_rounds"]):
    _rep3 = [BAL_POLICY, STAB_POLICY, EFF_POLICY]
    avail = {pol: REP[(pol, rnd_idx)] for pol in _rep3 if (pol, rnd_idx) in REP}
    if len(avail) < 2:
        continue
    desc = []
    for pol, d in avail.items():
        m = d["metrics"]
        desc.append(f"{pol}: OSR={m.osr:.3f}, WBI={m.wbi:.3f}")
    sft.append({
        "instruction": "동일 적재 시나리오에서 세 정책(BL/SF/EF)의 결과 차이를 안전 우선 vs 효율 우선 관점에서 설명하라.",
        "input": f"round={rnd_idx+1}, level={SEQ[rnd_idx]['level']}; " + "; ".join(desc),
        "output": "SF(안전 우선)는 무게균형·무게중심 안정성(WBI)이 상대적으로 높고, "
                  "EF(효율 우선)는 재취급 회피(OSR↓)·POD 그룹핑에 강점을 보이는 경향. "
                  "BL(통제군)은 두 목표를 동시에 학습하여 중간 수준의 균형을 보임.",
        "meta": {"round_id": rnd_idx+1, "type": "policy_comparison"},
    })

# ── (d) text2Cypher / NL2SQL 질문-쿼리 쌍 ──
t2c = [
    {"instruction": "다음 질문을 Cypher 쿼리로 변환하라.",
     "input": "SF 정책에서 재취급(overstow)이 발생한 슬롯은?",
     "output": "MATCH (s:Slot {policy:'SF'})-[:VIOLATES]->(c:Constraint {cons_id:'C_OVERSTOW'}) "
               "RETURN s.round_id, s.row, s.tier;",
     "meta": {"type": "text2cypher"}},
    {"instruction": "다음 질문을 Cypher 쿼리로 변환하라.",
     "input": "이 배정이 컬럼 무게 제약(SOLAS)을 위반하는가?",
     "output": "MATCH (s:Slot)-[:VIOLATES]->(c:Constraint {code:'SOLAS_VI'}) "
               "RETURN s.policy, s.round_id, s.row;",
     "meta": {"type": "text2cypher"}},
    {"instruction": "다음 질문을 SQL 쿼리로 변환하라.",
     "input": "정책별 평균 재취급률(OSR)을 구하라.",
     "output": "SELECT policy, AVG(osr) AS avg_osr FROM kpi GROUP BY policy;",
     "meta": {"type": "nl2sql"}},
    {"instruction": "다음 질문을 SQL 쿼리로 변환하라.",
     "input": "라운드별 컬럼 무게 위반 건수는?",
     "output": "SELECT round_id, SUM(n_col_wt_viol) AS viol FROM violation_log "
               "WHERE scope='SUMMARY' GROUP BY round_id;",
     "meta": {"type": "nl2sql"}},
]
sft.extend(t2c)

# ── 저장: JSONL (SFT 표준 포맷) + 타입별 통계 ──
sft_path = os.path.join(SLM_DIR, f"{CONFIG['experiment_name']}_sft_seed{GLOBAL_SEED}.jsonl")
with open(sft_path, "w", encoding="utf-8") as f:
    for rec in sft:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

from collections import Counter
type_counts = Counter(r["meta"].get("type") for r in sft)

print("═" * 70)
print("  🧠 SLM 파인튜닝 데이터셋 export — 슬라이드 ④")
print("═" * 70)
print(f"    · SFT 레코드 총 {len(sft)} 개  → {os.path.basename(sft_path)} (JSONL)")
for tp, n in type_counts.items():
    print(f"        - {tp:<18} {n} 개")
print()
print("  ── SFT 예시 (placement) ──")
ex = next((r for r in sft if r['meta'].get('type')=='placement'), sft[0])
print(f"    instruction: {ex['instruction']}")
print(f"    input      : {ex['input']}")
print(f"    output     : {ex['output'][:80]}...")

try:
    from google.colab import files
    files.download(sft_path)
    print()
    print("  ⬇️  Colab 다운로드 트리거 완료")
except Exception:
    print(f"  ℹ️  (비Colab) {SLM_DIR}")
