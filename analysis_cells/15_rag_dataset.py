# ══════════════════════════════════════════════════════════════════
#  ★ NEW (v14): RAG — 그림·설명(근거) 데이터셋 export  (슬라이드 ③)
#   · 검색·근거 제시용 비정형/시각 자료
#       (a) Bay Plan 시각화 이미지 (라운드별 PNG)
#       (b) 계획안 근거 설명 청크 (자연어, reward 분해 기반 자동 생성)
#       (c) 규정 코퍼스 스텁 (SOP/SOLAS/IMO/ISPS — 별도 문서, 인용용 placeholder)
#       (d) xAI 그라운딩 소스 = [RL계획 + Cypher검증 + 문서근거 + 운영수치] 병합 JSON
#   · ★ REP_CACHE(Cell 14에서 생성) 재사용 — Cell 14를 먼저 실행해야 함
# ══════════════════════════════════════════════════════════════════
import os, json, copy
import matplotlib.pyplot as plt

RAG_DIR = "/content/results/rag"
IMG_DIR = os.path.join(RAG_DIR, "bay_plan_images")
os.makedirs(IMG_DIR, exist_ok=True)

assert "REP_CACHE" in dir(), "REP_CACHE 없음 — Cell 14(RDB/LPG)를 먼저 실행하세요."
REP = REP_CACHE
SEQ = SEQ_CACHE

R15_LABELS = {
    "R3_overstow": "재취급(overstow) 회피", "R4_order": "POD 반출순서 정렬",
    "R5_weight_bal": "행간 무게 균형", "R6_cog": "무게중심(COG) 하강",
    "R8_pod_band": "POD 밴드 품질", "R10_tier_match": "동일 POD 수평 적재",
    "R11_wt_inversion": "무게 역전 방지", "R12_col_order": "컬럼 내 적재순서",
    "R13_tier_band": "Same-Tier 밴드 정렬",
}

# ── (a) Bay Plan 이미지 (정책 무관 표현 → SF 사용, 라운드별) ──
ref_pol = "SF" if ("SF", 0) in REP else next(iter(REP))[0]
img_paths = []
for rnd_idx in range(CONFIG["total_rounds"]):
    if (ref_pol, rnd_idx) not in REP:
        continue
    bp = REP[(ref_pol, rnd_idx)]["bay_plan"]
    pod_grid, wt_grid, stack_h = bp["pod_grid"], bp["wt_grid"], bp["stack_h"]
    n_rows, n_tiers = bp["n_rows"], bp["n_tiers"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
    pod_disp = np.full((n_tiers, n_rows), np.nan)
    wt_disp  = np.full((n_tiers, n_rows), np.nan)
    for rr in range(n_rows):
        for t in range(int(stack_h[rr])):
            pod_disp[n_tiers-1-t, rr] = pod_grid[rr, t]
            wt_disp[n_tiers-1-t, rr]  = wt_grid[rr, t]
    im0 = axes[0].imshow(pod_disp, cmap="tab10", vmin=1, vmax=6, aspect="auto")
    axes[0].set_title(f"POD distribution (R{rnd_idx+1})"); axes[0].set_xlabel("Row"); axes[0].set_ylabel("Tier (top→bottom)")
    im1 = axes[1].imshow(wt_disp, cmap="RdYlGn_r", aspect="auto")
    axes[1].set_title(f"Weight distribution MT (R{rnd_idx+1})"); axes[1].set_xlabel("Row")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    p = os.path.join(IMG_DIR, f"bayplan_{ref_pol}_R{rnd_idx+1}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    img_paths.append(p)

# ── (b) 근거 설명 청크 (reward 분해 → 자연어 근거) ──
def _rationale_chunks(pol, rnd_idx):
    d  = REP[(pol, rnd_idx)]; m = d["metrics"]; rc = m.reward_components
    chunks = []
    # reward 항목 상위 기여 → "왜 좋은가" 근거
    pos = sorted([(k, v) for k, v in rc.items() if v > 0 and k in R15_LABELS],
                 key=lambda x: -x[1])[:3]
    for k, v in pos:
        chunks.append(f"{R15_LABELS[k]} 우수 (보상 {v:+.1f}) — 해당 항목이 양(+)의 기여를 하여 계획 품질을 높임.")
    # KPI 기반 근거
    if m.osr < 0.05:
        chunks.append(f"재취급률(OSR) {m.osr:.3f}로 낮아 양하 시 불필요한 재취급이 거의 없음.")
    if m.wbi > 0.8:
        chunks.append(f"무게균형지수(WBI) {m.wbi:.3f}로 높아 행간 하중이 고르게 분산됨.")
    if m.psr > 0.7:
        chunks.append(f"POD 순수도(PSR) {m.psr:.3f} — 같은 목적항 컨테이너가 한 행에 그룹핑되어 반출순서 최적화에 유리.")
    return chunks

rows_chunk = []
for (pol, rnd_idx) in REP:
    for ci, txt in enumerate(_rationale_chunks(pol, rnd_idx)):
        rows_chunk.append({
            "chunk_id": f"{pol}_R{rnd_idx+1}_c{ci}",
            "policy": pol, "round_id": rnd_idx+1,
            "source_type": "RL_RATIONALE", "text": txt,
        })
# 규정 코퍼스 스텁 (별도 문서 — 인용 placeholder)
reg_stub = [
    ("SOLAS_VI", "선박 적재 시 컬럼별 허용 하중을 초과하지 않아야 한다 (복원성 확보)."),
    ("IMO_STABILITY", "무거운 화물은 하부 Tier에 적재하여 무게중심을 낮춘다 (heavy-down)."),
    ("SOP_STOWAGE", "동일 POD 컨테이너는 가능한 그룹핑하여 반출 효율을 높인다."),
    ("ISPS", "위험물 컨테이너는 격리 규칙에 따라 배치한다."),
]
for code, txt in reg_stub:
    rows_chunk.append({"chunk_id": f"REG_{code}", "policy": "", "round_id": None,
                       "source_type": "REGULATION", "text": txt})

import pandas as pd
df_chunk = pd.DataFrame(rows_chunk)
df_chunk.to_csv(os.path.join(RAG_DIR, "rationale_chunks.csv"),
                index=False, encoding="utf-8-sig")
# 임베딩 친화 JSONL (id, text, metadata)
with open(os.path.join(RAG_DIR, "rag_chunks.jsonl"), "w", encoding="utf-8") as f:
    for r in rows_chunk:
        f.write(json.dumps({"id": r["chunk_id"], "text": r["text"],
                            "metadata": {"policy": r["policy"], "round_id": r["round_id"],
                                         "source_type": r["source_type"]}},
                           ensure_ascii=False) + "\n")

# ── (d) xAI 그라운딩 소스 (RL계획 + Cypher검증 + 문서근거 + 운영수치) ──
grounding = []
for (pol, rnd_idx), d in REP.items():
    m = d["metrics"]
    grounding.append({
        "policy": pol, "round_id": rnd_idx+1,
        "rl_plan": {"reward": round(d["reward"],2),
                    "bay_shape": [d["bay_plan"]["n_rows"], d["bay_plan"]["n_tiers"]]},
        "cypher_facts": {"n_overstow": m.n_overstow, "n_col_wt_viol": m.n_col_wt_viol},
        "doc_refs": [c[0] for c in reg_stub],
        "ops_metrics": {"osr": round(m.osr,4), "wbi": round(m.wbi,4),
                        "psr": round(m.psr,4), "cwvr": round(m.cwvr,4)},
        "rationale": _rationale_chunks(pol, rnd_idx),
    })
with open(os.path.join(RAG_DIR, "xai_grounding.json"), "w", encoding="utf-8") as f:
    json.dump(grounding, f, ensure_ascii=False, indent=2)

print("═" * 70)
print("  📚 RAG 데이터셋 export — 슬라이드 ③")
print("═" * 70)
print(f"    · Bay Plan 이미지     {len(img_paths)} 장  → {IMG_DIR}")
print(f"    · 근거/규정 청크       {len(df_chunk)} 개  → rationale_chunks.csv / rag_chunks.jsonl")
print(f"    · xAI 그라운딩 소스    {len(grounding)} 건  → xai_grounding.json")

import zipfile
rag_zip = f"/content/results/{CONFIG['experiment_name']}_RAG_seed{GLOBAL_SEED}.zip"
with zipfile.ZipFile(rag_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in img_paths: zf.write(p, arcname=f"images/{os.path.basename(p)}")
    for fn in ["rationale_chunks.csv", "rag_chunks.jsonl", "xai_grounding.json"]:
        zf.write(os.path.join(RAG_DIR, fn), arcname=fn)
print(f"  📦 RAG 번들: {rag_zip}")

# 후속 SLM 셀 재사용
RAG_CHUNKS = rows_chunk
RAG_GROUNDING = grounding

try:
    from google.colab import files
    files.download(rag_zip)
    print("  ⬇️  Colab 다운로드 트리거 완료")
except Exception:
    print(f"  ℹ️  (비Colab) {RAG_DIR}")
