# ══════════════════════════════════════════════════════════════════
#  ★ NEW (v14): RDB(관계형) + LPG(Neo4j 지식그래프) 통합 데이터셋 export
#   · 슬라이드 ①RDB / ②LPG 설계를 그대로 코드화
#   · 공통 소스: 각 (정책 × 라운드)의 대표 eval 에피소드(ep0)를 재평가하여
#       bay_plan + StowageMetrics(violation, reward_components R1~R15) 동시 확보
#   · vessel/voyage 는 단일베이 실험이므로 상수 메타로 부여
# ──────────────────────────────────────────────────────────────────
#  ▣ RDB (관계형) — 슬라이드 ①
#     - slot_assignment : vessel,voyage,bay,row,tier,container_id,POD,weight_MT,loading_order
#     - kpi             : VPR,OSR,WBI,PSR,CWVR,reward (정책×라운드)
#     - violation_log   : n_overstow,n_col_wt_viol,n_empty_rows + 위반 슬롯 위치
#     - reward_decomp   : R1~R15 항목별 점수 (정책×라운드)
#     - agent_run       : 정책/seed/생성시각/학습통계 (DecisionLog 베이스)
#     - nl2sql_metrics  : NL2SQL 질의 대상 운영수치 long 테이블
#  ▣ LPG (Neo4j) — 슬라이드 ②
#     - 노드: Vessel,Bay,Row,Tier,Slot,Container,Port(POD),Policy,Constraint
#     - 관계: Bay-Row-Tier-Slot 계층, Container-[:ASSIGNED_TO]->Slot,
#             Container-[:STACKED_ON]->Container(오버스토우), Container-[:HAS_POD]->Port,
#             Slot-[:VIOLATES]->Constraint, Policy-[:ACHIEVED]->...
# ══════════════════════════════════════════════════════════════════
import os, csv, copy, datetime
import pandas as pd

RDB_DIR = "/content/results/rdb"
KG_DIR  = "/content/results/neo4j_kg"
os.makedirs(RDB_DIR, exist_ok=True)
os.makedirs(KG_DIR,  exist_ok=True)

# 단일베이 실험 메타 (RDB vessel/voyage, LPG Vessel/Bay 노드용)
VESSEL_ID  = "VSL_SINGLEBAY"
VOYAGE_ID  = f"VOY_{GLOBAL_SEED}"
BAY_ID     = "BAY_01"

# ── 대표 에피소드 재평가: (정책×라운드) → (bay_plan, metrics) ──
#    _evaluate_episode_full 은 BASE 보상 env 로 평가 (정책 차이만 측정)
EVAL_BASE_CFG = copy.deepcopy(CONFIG)
EVAL_BASE_CFG["rw"] = dict(CONFIG["rw"])     # BASE 가중치

def _eval_rep(pol_res, rnd_idx):
    """라운드 대표 eval ep0 재평가 → (reward, metrics, bay_plan)."""
    lv = pol_res["levels"][rnd_idx]
    rep_seed = GLOBAL_SEED + 0 * 17
    return _evaluate_episode_full(pol_res["trained_model"], lv, EVAL_BASE_CFG, rep_seed)

# (정책,라운드) → eval 결과 캐시
REP = {}
for pol_name, res in all_results.items():
    if res.get("trained_model") is None:
        continue
    for rnd_idx in range(CONFIG["total_rounds"]):
        r, m, bp = _eval_rep(res, rnd_idx)
        REP[(pol_name, rnd_idx)] = {"reward": r, "metrics": m, "bay_plan": bp}

# 입력 컨테이너 시퀀스(정책 무관) 캐시 — 라운드별
SEQ = {}
for rnd_idx in range(CONFIG["total_rounds"]):
    lv = next(iter(all_results.values()))["levels"][rnd_idx]
    env = make_env(level=lv, config=CONFIG); env.reset(seed=GLOBAL_SEED + 0*17)
    SEQ[rnd_idx] = {"pod": env.ctns_pod.copy(), "wt": env.ctns_wt.copy(), "level": lv}

R15_KEYS = ["R1_valid","R2_stack_full","R3_overstow","R4_order","R5_weight_bal",
            "R6_cog","R7_completion","R8_pod_band","R9_col_wt","R10_tier_match",
            "R11_wt_inversion","R12_col_order","R13_tier_band","R14_empty_row","R15_vstack_pod"]

# ════════════════════════════════════════════════════════════════
#  PART A — RDB (관계형) 테이블
# ════════════════════════════════════════════════════════════════

def _slot_container_id(pol, rnd, row, tier):
    return f"{pol}_R{rnd+1}_r{row}_t{tier}"

# ── A1. slot_assignment ──
rows_slot = []
for (pol_name, rnd_idx), d in REP.items():
    bp = d["bay_plan"]; lv = SEQ[rnd_idx]["level"]
    pod_grid, wt_grid, stack_h = bp["pod_grid"], bp["wt_grid"], bp["stack_h"]
    for rr in range(bp["n_rows"]):
        for t in range(int(stack_h[rr])):
            pod = int(pod_grid[rr, t])
            rows_slot.append({
                "vessel":        VESSEL_ID,
                "voyage":        VOYAGE_ID,
                "bay":           BAY_ID,
                "policy":        pol_name,
                "round_id":      rnd_idx + 1,
                "level":         lv,
                "row":           rr,
                "tier":          t,
                "container_id":  _slot_container_id(pol_name, rnd_idx, rr, t),
                "pod_id":        pod,
                "pod_name":      POD_NAMES.get(pod, "NA"),
                "weight_mt":     round(float(wt_grid[rr, t]), 2),
                "loading_order": None,    # bay_plan 은 위치만 — 입력순번은 fact_container 측
                "is_bottom":     int(t == 0),
                "is_top":        int(t == int(stack_h[rr]) - 1),
            })
df_slot = pd.DataFrame(rows_slot)

# ── A2. kpi ──
rows_kpi = []
for (pol_name, rnd_idx), d in REP.items():
    m = d["metrics"]
    rows_kpi.append({
        "policy": pol_name, "round_id": rnd_idx + 1, "level": SEQ[rnd_idx]["level"],
        "reward": round(d["reward"], 3),
        "vpr": round(m.vpr, 4), "osr": round(m.osr, 4), "wbi": round(m.wbi, 4),
        "psr": round(m.psr, 4), "cwvr": round(m.cwvr, 4),
        "n_empty_rows": m.n_empty_rows,
    })
df_kpi = pd.DataFrame(rows_kpi)

# ── A3. violation_log (집계 + 위반 슬롯 위치) ──
rows_viol = []
for (pol_name, rnd_idx), d in REP.items():
    m  = d["metrics"]; bp = d["bay_plan"]
    pod_grid, wt_grid, stack_h = bp["pod_grid"], bp["wt_grid"], bp["stack_h"]
    # 집계 요약 행
    rows_viol.append({
        "policy": pol_name, "round_id": rnd_idx + 1, "scope": "SUMMARY",
        "row": None, "tier": None, "viol_type": "AGGREGATE",
        "n_overstow": m.n_overstow, "n_col_wt_viol": m.n_col_wt_viol,
        "n_empty_rows": m.n_empty_rows, "detail": "",
    })
    # 슬롯별 overstow 위치: 같은 row 에서 아래 pod_id < 위 pod_id (위가 먼저 하역)
    for rr in range(bp["n_rows"]):
        h = int(stack_h[rr])
        for t in range(1, h):
            lower, upper = int(pod_grid[rr, t-1]), int(pod_grid[rr, t])
            if lower < upper:   # 아래가 더 가까운 항(작은 pod_id) → 위가 막아서 재취급
                rows_viol.append({
                    "policy": pol_name, "round_id": rnd_idx + 1, "scope": "SLOT",
                    "row": rr, "tier": t, "viol_type": "OVERSTOW",
                    "n_overstow": 1, "n_col_wt_viol": 0, "n_empty_rows": 0,
                    "detail": f"upper POD{upper}({POD_NAMES[upper]}) blocks "
                              f"lower POD{lower}({POD_NAMES[lower]})",
                })
    # 컬럼 무게 위반 위치
    for rr in range(bp["n_rows"]):
        col_wt = float(wt_grid[rr, :int(stack_h[rr])].sum())
        if col_wt > MAX_COL_WT:
            rows_viol.append({
                "policy": pol_name, "round_id": rnd_idx + 1, "scope": "SLOT",
                "row": rr, "tier": None, "viol_type": "COL_WT",
                "n_overstow": 0, "n_col_wt_viol": 1, "n_empty_rows": 0,
                "detail": f"col_wt={col_wt:.1f} > MAX {MAX_COL_WT:.0f} MT",
            })
df_viol = pd.DataFrame(rows_viol)

# ── A4. reward_decomp (R1~R15) ──
rows_rd = []
for (pol_name, rnd_idx), d in REP.items():
    rc = d["metrics"].reward_components
    row = {"policy": pol_name, "round_id": rnd_idx + 1, "level": SEQ[rnd_idx]["level"]}
    for k in R15_KEYS:
        row[k] = round(float(rc.get(k, 0.0)), 3)
    row["reward_total"] = round(d["reward"], 3)
    rows_rd.append(row)
df_rd = pd.DataFrame(rows_rd)

# ── A5. agent_run / DecisionLog ──
now_iso = datetime.datetime.now().isoformat(timespec="seconds")
rows_run = []
for pol_name, res in all_results.items():
    if res.get("trained_model") is None:
        continue
    for prm in res["per_round_metrics"]:
        rows_run.append({
            "run_id":       f"{CONFIG['experiment_name']}_{pol_name}_R{prm['round']}",
            "policy":       pol_name,
            "model_version":"v16",
            "seed":         GLOBAL_SEED,
            "round_id":     prm["round"],
            "level":        prm["level"],
            "created_at":   now_iso,
            "episodes":     prm.get("actual_episodes"),
            "steps":        prm.get("actual_steps"),
            "train_sec":    round(prm.get("train_seconds", 0.0), 1),
            "approval_status": "PENDING",   # 운영자 승인/반려/수정 기록용 (초기값)
            "operator":     "",
            "note":         "",
        })
df_run = pd.DataFrame(rows_run)

# ── A6. nl2sql_metrics (운영수치 long) ──
rows_nl = []
for (pol_name, rnd_idx), d in REP.items():
    m = d["metrics"]
    base = {"policy": pol_name, "round_id": rnd_idx + 1}
    for metric, val in [("n_dangerous_pods", None),  # 단일베이 실험엔 DG 미구분 → NULL
                        ("osr", round(m.osr,4)), ("vpr", round(m.vpr,4)),
                        ("wbi", round(m.wbi,4)), ("psr", round(m.psr,4)),
                        ("cwvr", round(m.cwvr,4)), ("n_overstow", m.n_overstow),
                        ("n_col_wt_viol", m.n_col_wt_viol), ("n_empty_rows", m.n_empty_rows),
                        ("throughput_containers", int(m.n_total))]:
        rows_nl.append({**base, "metric_name": metric, "metric_value": val})
df_nl = pd.DataFrame(rows_nl)

rdb_tables = {
    "slot_assignment": df_slot, "kpi": df_kpi, "violation_log": df_viol,
    "reward_decomp": df_rd, "agent_run": df_run, "nl2sql_metrics": df_nl,
}

# 저장: CSV + 통합 Excel
for name, df in rdb_tables.items():
    df.to_csv(os.path.join(RDB_DIR, f"{name}.csv"), index=False, encoding="utf-8-sig")
rdb_xlsx = os.path.join(RDB_DIR, f"{CONFIG['experiment_name']}_RDB_seed{GLOBAL_SEED}.xlsx")
with pd.ExcelWriter(rdb_xlsx, engine="openpyxl") as xw:
    for name, df in rdb_tables.items():
        df.to_excel(xw, sheet_name=name[:31], index=False)

print("═" * 70)
print("  🗄️  RDB(관계형) 데이터셋 export — 슬라이드 ①")
print("═" * 70)
for name, df in rdb_tables.items():
    print(f"    · {name:<16} rows={len(df):>5}  cols={len(df.columns):>2}")
print(f"  통합 Excel: {rdb_xlsx}")

# ════════════════════════════════════════════════════════════════
#  PART B — LPG (Neo4j 지식그래프) 노드/관계 CSV  (슬라이드 ②)
# ════════════════════════════════════════════════════════════════

def _dump_kg(rows, fname):
    pd.DataFrame(rows).to_csv(os.path.join(KG_DIR, fname),
                              index=False, encoding="utf-8-sig")
    return len(rows)

# ── 노드 ──
node_vessel = [{"vessel_id:ID(Vessel)": VESSEL_ID, "voyage": VOYAGE_ID, ":LABEL": "Vessel"}]
node_bay    = [{"bay_id:ID(Bay)": BAY_ID, "vessel": VESSEL_ID, ":LABEL": "Bay"}]
node_port   = [{"pod_id:ID(Port)": pid, "pod_name": nm, "distance_rank:int": pid, ":LABEL": "Port"}
               for pid, nm in POD_NAMES.items()]
node_policy = [{"policy:ID(Policy)": pn, "desc": POLICIES[pn]["desc"], ":LABEL": "Policy"}
               for pn in all_results if all_results[pn].get("trained_model") is not None]

# Constraint 노드 (IMO/SOLAS 기반) — text2Cypher 위반 검증용
node_constraint = [
    {"cons_id:ID(Constraint)": "C_OVERSTOW", "code": "STOWAGE",
     "rule": "POD discharge order must not be blocked by later-POD on top",
     "source": "Operational", ":LABEL": "Constraint"},
    {"cons_id:ID(Constraint)": "C_COL_WT", "code": "SOLAS_VI",
     "rule": f"Column weight <= {MAX_COL_WT:.0f} MT",
     "source": "SOLAS", ":LABEL": "Constraint"},
    {"cons_id:ID(Constraint)": "C_COG", "code": "IMO_STABILITY",
     "rule": "Heavy containers lower (heavy-down) for COG/stability",
     "source": "IMO", ":LABEL": "Constraint"},
]

# Row / Tier / Slot / Container 노드 + 관계 (정책×라운드별)
node_row, node_tier, node_slot, node_container = [], [], [], []
rel_bay_row, rel_row_tier, rel_tier_slot = [], [], []
rel_assigned, rel_has_pod, rel_stacked, rel_violates = [], [], [], []
rel_achieved = []
_seen_row, _seen_tier = set(), set()

for (pol_name, rnd_idx), d in REP.items():
    bp = d["bay_plan"]; m = d["metrics"]
    pod_grid, wt_grid, stack_h = bp["pod_grid"], bp["wt_grid"], bp["stack_h"]
    n_rows = bp["n_rows"]
    tag = f"{pol_name}_R{rnd_idx+1}"
    for rr in range(n_rows):
        row_id = f"{tag}_r{rr}"
        node_row.append({"row_id:ID(Row)": row_id, "policy": pol_name,
                         "round_id:int": rnd_idx+1, "row:int": rr, ":LABEL": "Row"})
        rel_bay_row.append({":START_ID(Bay)": BAY_ID, ":END_ID(Row)": row_id, ":TYPE": "HAS_ROW"})
        prev_cid = None
        h = int(stack_h[rr])
        for t in range(h):
            tier_id = f"{row_id}_t{t}"
            slot_id = f"{tier_id}_slot"
            cid     = _slot_container_id(pol_name, rnd_idx, rr, t)
            pod     = int(pod_grid[rr, t])
            w       = round(float(wt_grid[rr, t]), 2)
            node_tier.append({"tier_id:ID(Tier)": tier_id, "row_id": row_id,
                              "tier:int": t, ":LABEL": "Tier"})
            node_slot.append({"slot_id:ID(Slot)": slot_id, "policy": pol_name,
                              "round_id:int": rnd_idx+1, "row:int": rr, "tier:int": t,
                              "is_bottom:int": int(t==0), "is_top:int": int(t==h-1),
                              ":LABEL": "Slot"})
            node_container.append({"container_id:ID(Container)": cid, "policy": pol_name,
                                   "round_id:int": rnd_idx+1, "pod_id:int": pod,
                                   "weight_mt:float": w, "row:int": rr, "tier:int": t,
                                   ":LABEL": "Container"})
            rel_row_tier.append({":START_ID(Row)": row_id, ":END_ID(Tier)": tier_id, ":TYPE": "HAS_TIER"})
            rel_tier_slot.append({":START_ID(Tier)": tier_id, ":END_ID(Slot)": slot_id, ":TYPE": "HAS_SLOT"})
            rel_assigned.append({":START_ID(Container)": cid, ":END_ID(Slot)": slot_id, ":TYPE": "ASSIGNED_TO"})
            rel_has_pod.append({":START_ID(Container)": cid, ":END_ID(Port)": pod, ":TYPE": "HAS_POD"})
            # STACKED_ON: 위 컨테이너 → 아래 컨테이너 (적재순서/재취급)
            if prev_cid is not None:
                lower_pod = int(pod_grid[rr, t-1])
                is_ov = int(lower_pod < pod)   # 아래가 먼저 하역해야 하는데 위가 막음
                rel_stacked.append({":START_ID(Container)": cid, ":END_ID(Container)": prev_cid,
                                    "is_overstow:int": is_ov, ":TYPE": "STACKED_ON"})
            prev_cid = cid
            # Slot-[:VIOLATES]->Constraint (overstow)
            if t >= 1 and int(pod_grid[rr, t-1]) < pod:
                rel_violates.append({":START_ID(Slot)": slot_id,
                                     ":END_ID(Constraint)": "C_OVERSTOW", ":TYPE": "VIOLATES"})
        # 컬럼 무게 위반 → 최상단 슬롯이 위반
        col_wt = float(wt_grid[rr, :h].sum())
        if col_wt > MAX_COL_WT and h > 0:
            rel_violates.append({":START_ID(Slot)": f"{row_id}_t{h-1}_slot",
                                 ":END_ID(Constraint)": "C_COL_WT", ":TYPE": "VIOLATES"})

    # Policy-[:ACHIEVED]->Bay (라운드 KPI as 관계 속성)
    rel_achieved.append({":START_ID(Policy)": pol_name, ":END_ID(Bay)": BAY_ID,
                         "round_id:int": rnd_idx+1, "reward:float": round(d["reward"],3),
                         "osr:float": round(m.osr,4), "wbi:float": round(m.wbi,4),
                         "vpr:float": round(m.vpr,4), "psr:float": round(m.psr,4),
                         "cwvr:float": round(m.cwvr,4), ":TYPE": "ACHIEVED"})

kg_nodes = {
    "Vessel.csv": node_vessel, "Bay.csv": node_bay, "Row.csv": node_row,
    "Tier.csv": node_tier, "Slot.csv": node_slot, "Container.csv": node_container,
    "Port.csv": node_port, "Policy.csv": node_policy, "Constraint.csv": node_constraint,
}
kg_rels = {
    "REL_HAS_ROW.csv": rel_bay_row, "REL_HAS_TIER.csv": rel_row_tier,
    "REL_HAS_SLOT.csv": rel_tier_slot, "REL_ASSIGNED_TO.csv": rel_assigned,
    "REL_HAS_POD.csv": rel_has_pod, "REL_STACKED_ON.csv": rel_stacked,
    "REL_VIOLATES.csv": rel_violates, "REL_ACHIEVED.csv": rel_achieved,
}

print()
print("═" * 70)
print("  🕸️  LPG(Neo4j) 지식그래프 데이터셋 export — 슬라이드 ②")
print("═" * 70)
print("  ── NODES ──")
for fn, rows in kg_nodes.items():
    print(f"    · {fn:<16} {_dump_kg(rows, fn):>5} nodes")
print("  ── RELATIONSHIPS ──")
for fn, rows in kg_rels.items():
    print(f"    · {fn:<22} {_dump_kg(rows, fn):>5} rels")

# Cypher import 스크립트 (LOAD CSV) + text2Cypher 검증 쿼리
cypher = r"""// ════ Neo4j import (LOAD CSV) — 단일베이 적재계획 지식그래프 ════
CREATE CONSTRAINT vessel_id IF NOT EXISTS FOR (n:Vessel) REQUIRE n.vessel_id IS UNIQUE;
CREATE CONSTRAINT bay_id    IF NOT EXISTS FOR (n:Bay) REQUIRE n.bay_id IS UNIQUE;
CREATE CONSTRAINT row_id    IF NOT EXISTS FOR (n:Row) REQUIRE n.row_id IS UNIQUE;
CREATE CONSTRAINT tier_id   IF NOT EXISTS FOR (n:Tier) REQUIRE n.tier_id IS UNIQUE;
CREATE CONSTRAINT slot_id   IF NOT EXISTS FOR (n:Slot) REQUIRE n.slot_id IS UNIQUE;
CREATE CONSTRAINT cont_id   IF NOT EXISTS FOR (n:Container) REQUIRE n.container_id IS UNIQUE;
CREATE CONSTRAINT port_id   IF NOT EXISTS FOR (n:Port) REQUIRE n.pod_id IS UNIQUE;
CREATE CONSTRAINT policy_id IF NOT EXISTS FOR (n:Policy) REQUIRE n.policy IS UNIQUE;
CREATE CONSTRAINT cons_id   IF NOT EXISTS FOR (n:Constraint) REQUIRE n.cons_id IS UNIQUE;

LOAD CSV WITH HEADERS FROM 'file:///Vessel.csv' AS r CREATE (:Vessel {vessel_id:r.`vessel_id:ID(Vessel)`, voyage:r.voyage});
LOAD CSV WITH HEADERS FROM 'file:///Bay.csv' AS r CREATE (:Bay {bay_id:r.`bay_id:ID(Bay)`, vessel:r.vessel});
LOAD CSV WITH HEADERS FROM 'file:///Row.csv' AS r CREATE (:Row {row_id:r.`row_id:ID(Row)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), row:toInteger(r.`row:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Tier.csv' AS r CREATE (:Tier {tier_id:r.`tier_id:ID(Tier)`, row_id:r.row_id, tier:toInteger(r.`tier:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Slot.csv' AS r CREATE (:Slot {slot_id:r.`slot_id:ID(Slot)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), row:toInteger(r.`row:int`), tier:toInteger(r.`tier:int`), is_bottom:toInteger(r.`is_bottom:int`), is_top:toInteger(r.`is_top:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Container.csv' AS r CREATE (:Container {container_id:r.`container_id:ID(Container)`, policy:r.policy, round_id:toInteger(r.`round_id:int`), pod_id:toInteger(r.`pod_id:int`), weight_mt:toFloat(r.`weight_mt:float`), row:toInteger(r.`row:int`), tier:toInteger(r.`tier:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Port.csv' AS r CREATE (:Port {pod_id:toInteger(r.`pod_id:ID(Port)`), pod_name:r.pod_name, distance_rank:toInteger(r.`distance_rank:int`)});
LOAD CSV WITH HEADERS FROM 'file:///Policy.csv' AS r CREATE (:Policy {policy:r.`policy:ID(Policy)`, desc:r.desc});
LOAD CSV WITH HEADERS FROM 'file:///Constraint.csv' AS r CREATE (:Constraint {cons_id:r.`cons_id:ID(Constraint)`, code:r.code, rule:r.rule, source:r.source});

LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_ROW.csv' AS r MATCH (a:Bay {bay_id:r.`:START_ID(Bay)`}),(b:Row {row_id:r.`:END_ID(Row)`}) CREATE (a)-[:HAS_ROW]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_TIER.csv' AS r MATCH (a:Row {row_id:r.`:START_ID(Row)`}),(b:Tier {tier_id:r.`:END_ID(Tier)`}) CREATE (a)-[:HAS_TIER]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_SLOT.csv' AS r MATCH (a:Tier {tier_id:r.`:START_ID(Tier)`}),(b:Slot {slot_id:r.`:END_ID(Slot)`}) CREATE (a)-[:HAS_SLOT]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_ASSIGNED_TO.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Slot {slot_id:r.`:END_ID(Slot)`}) CREATE (a)-[:ASSIGNED_TO]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_HAS_POD.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Port {pod_id:toInteger(r.`:END_ID(Port)`)}) CREATE (a)-[:HAS_POD]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_STACKED_ON.csv' AS r MATCH (a:Container {container_id:r.`:START_ID(Container)`}),(b:Container {container_id:r.`:END_ID(Container)`}) CREATE (a)-[:STACKED_ON {is_overstow:toInteger(r.`is_overstow:int`)}]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_VIOLATES.csv' AS r MATCH (a:Slot {slot_id:r.`:START_ID(Slot)`}),(b:Constraint {cons_id:r.`:END_ID(Constraint)`}) CREATE (a)-[:VIOLATES]->(b);
LOAD CSV WITH HEADERS FROM 'file:///REL_ACHIEVED.csv' AS r MATCH (a:Policy {policy:r.`:START_ID(Policy)`}),(b:Bay {bay_id:r.`:END_ID(Bay)`}) CREATE (a)-[:ACHIEVED {round_id:toInteger(r.`round_id:int`), reward:toFloat(r.`reward:float`), osr:toFloat(r.`osr:float`), wbi:toFloat(r.`wbi:float`), vpr:toFloat(r.`vpr:float`), psr:toFloat(r.`psr:float`), cwvr:toFloat(r.`cwvr:float`)}]->(b);

// ════ text2Cypher 검증 쿼리 예시 ════
// Q. "SF 정책에서 재취급(overstow)이 발생한 슬롯은?"
// MATCH (s:Slot {policy:'SF'})-[:VIOLATES]->(c:Constraint {cons_id:'C_OVERSTOW'})
// RETURN s.round_id, s.row, s.tier;
// Q. "이 배정이 컬럼 무게 제약(SOLAS)을 위반하는가?"
// MATCH (s:Slot)-[:VIOLATES]->(c:Constraint {code:'SOLAS_VI'}) RETURN s.policy, s.round_id, s.row;
// Q. "오버스토우 관계 체인 (Container STACKED_ON, is_overstow=1)"
// MATCH (up:Container)-[r:STACKED_ON {is_overstow:1}]->(down:Container)
// RETURN up.policy, up.row, up.tier, up.pod_id AS upper_pod, down.pod_id AS lower_pod;
"""
with open(os.path.join(KG_DIR, "import_cypher.cypher"), "w", encoding="utf-8") as f:
    f.write(cypher)
print(f"  📜 Cypher: {os.path.join(KG_DIR, 'import_cypher.cypher')}")

# ── 다운로드 번들 ──
import zipfile
bundle = f"/content/results/{CONFIG['experiment_name']}_RDB_LPG_seed{GLOBAL_SEED}.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
    for name in rdb_tables:
        zf.write(os.path.join(RDB_DIR, f"{name}.csv"), arcname=f"rdb/{name}.csv")
    zf.write(rdb_xlsx, arcname=f"rdb/{os.path.basename(rdb_xlsx)}")
    for fn in list(kg_nodes) + list(kg_rels):
        zf.write(os.path.join(KG_DIR, fn), arcname=f"neo4j_kg/{fn}")
    zf.write(os.path.join(KG_DIR, "import_cypher.cypher"), arcname="neo4j_kg/import_cypher.cypher")
print()
print(f"  📦 통합 번들(RDB+LPG): {bundle}")

# 후속 셀(RAG/SLM)에서 재사용하도록 전역 노출
RDB_TABLES = rdb_tables
REP_CACHE  = REP
SEQ_CACHE  = SEQ

try:
    from google.colab import files
    files.download(bundle)
    print("  ⬇️  Colab 다운로드 트리거 완료")
except Exception:
    print(f"  ℹ️  (비Colab) RDB={RDB_DIR}  KG={KG_DIR}")
