# 종합 비교표 + 사전 가설 검증 (★ v13: 3-way 확장)
print("═" * 110)
print("  TABLE: 6 policies — Round-wise Performance with Phase Info")
print("═" * 110)
print(f"  {'Policy':<8} {'Round':<6} {'Level':<6} {'Phase':<6} {'Mode':<18} "
      f"{'Reward':>9} {'OSR':>7} {'VPR':>7} {'WBI':>7} {'PSR':>7} {'CWVR':>7}")
print("  " + "─" * 110)

for pn, res in all_results.items():
    for rnd in range(CONFIG["total_rounds"]):
        pm = res["per_round_metrics"][rnd]
        pi = res["phase_info"][rnd]
        marker = "★" if rnd == CONFIG["total_rounds"] - 1 else " "
        print(f"  {pn:<8} {rnd+1:<6} Lv{pm['level']:<4} {pi['phase']:<6} "
              f"{pi['mode']:<18} {pm['avg_reward']:>+8.2f} "
              f"{pm['avg_osr']:>6.3f} {pm['avg_vpr']:>6.3f} "
              f"{pm['avg_wbi']:>6.3f} {pm['avg_psr']:>6.3f} "
              f"{pm.get('avg_cwvr', 0):>6.3f} {marker}")
    print("  " + "─" * 110)


# ── 일반화된 가설 검증 함수 ──
def hyp_check_pair(name, val_a, val_b, label_a, label_b, predicted, direction):
    """두 정책 간 비교.
    direction: 'high' = 큰 값이 우위, 'low' = 작은 값이 우위
    """
    if direction == "high":
        actual = label_a if val_a > val_b else label_b
    else:
        actual = label_a if val_a < val_b else label_b
    diff = abs(val_a - val_b)
    ok = "✅" if actual == predicted else "❌"
    return (f"  {ok} {name:<34} 가설={predicted:<3} 실제={actual:<3}  "
            f"{label_a}={val_a:.4f}  {label_b}={val_b:.4f}  Δ={diff:.4f}")


# ── 가설 검증 (Lv4 final round) ──
print()
print("═" * 110)
print("  사전 등록 가설 검증 (Final Round Lv4)")
print("═" * 110)

# 가설검증 대표 정책: 안정=SF_9_EF_1(STAB), 효율=EF_9_SF_1(EFF), 균형=BL_SF_5_EF_5(BAL)
sf_pm = all_results[STAB_POLICY]["per_round_metrics"][-1]
ef_pm = all_results[EFF_POLICY]["per_round_metrics"][-1]
bl_pm = all_results[BAL_POLICY]["per_round_metrics"][-1]

# H1~H4: SF vs EF
print(f"  ── H1~H4: {STAB_POLICY}(안정) vs {EFF_POLICY}(효율) ──")
print(hyp_check_pair("H1: WBI (Stability) ↑",    sf_pm["avg_wbi"],  ef_pm["avg_wbi"],  "STAB", "EFF", "STAB", "high"))
print(hyp_check_pair("H2: OSR (Efficiency) ↓",   sf_pm["avg_osr"],  ef_pm["avg_osr"],  "STAB", "EFF", "EFF", "low"))
print(hyp_check_pair("    CWVR (Safety) ↓",      sf_pm.get("avg_cwvr",0), ef_pm.get("avg_cwvr",0), "STAB", "EFF", "STAB", "low"))
print(hyp_check_pair("    PSR ↓ (Eff 영향)",     sf_pm["avg_psr"],  ef_pm["avg_psr"],  "STAB", "EFF", "EFF", "low"))
print(hyp_check_pair("    VPR (Procedural)",     sf_pm["avg_vpr"],  ef_pm["avg_vpr"],  "STAB", "EFF", "—",  "high"))

# H3: Trade-off 비대칭
wbi_advantage  = sf_pm["avg_wbi"] - ef_pm["avg_wbi"]
osr_advantage  = ef_pm["avg_osr"] - sf_pm["avg_osr"]   # EF.OSR이 작으면 양수 → EF 우위 magnitude
print()
print(f"  H3 (Trade-off 비대칭): SF의 WBI 우위 ({wbi_advantage:+.4f}) vs "
      f"EF의 OSR 우위 ({osr_advantage:+.4f})")
if abs(wbi_advantage) > abs(osr_advantage):
    print(f"      → |WBI Δ| > |OSR Δ| : Stability priority가 더 robust ✓")
else:
    print(f"      → |OSR Δ| > |WBI Δ| : Efficiency priority가 더 robust")

# ★ v13 NEW: H5a, H5b — vs Baseline
if bl_pm is not None:
    print()
    print("  ── ★ H5: vs Baseline (우선순위 부여 효과) ──")
    print(hyp_check_pair("H5a: STAB WBI > BAL WBI",
                         sf_pm["avg_wbi"], bl_pm["avg_wbi"], "STAB", "BAL", "STAB", "high"))
    print(hyp_check_pair("H5b: EFF OSR < BAL OSR",
                         ef_pm["avg_osr"], bl_pm["avg_osr"], "EFF", "BAL", "EFF", "low"))
    sf_vs_bl_wbi = sf_pm["avg_wbi"] - bl_pm["avg_wbi"]
    ef_vs_bl_osr = bl_pm["avg_osr"] - ef_pm["avg_osr"]
    print()
    print(f"  H5 효과 크기:")
    print(f"      H5a Δ = SF.WBI − BL.WBI = {sf_vs_bl_wbi:+.4f}  (threshold +0.02)")
    print(f"      H5b Δ = BL.OSR − EF.OSR = {ef_vs_bl_osr:+.4f}  (threshold +0.02)")
    if sf_vs_bl_wbi > 0.02 and ef_vs_bl_osr > 0.02:
        print(f"      → 양 정책 모두 BL 대비 자기 영역 KPI 향상 ✓ (lex priority 유효)")
    elif sf_vs_bl_wbi > 0.02:
        print(f"      → SF만 BL 대비 향상 — Stability priority만 유효")
    elif ef_vs_bl_osr > 0.02:
        print(f"      → EF만 BL 대비 향상 — Efficiency priority만 유효")
    else:
        print(f"      → 두 정책 모두 BL 대비 임계값 미달 — priority 효과 한정적")


# ── 3-way Lv4 종합 ──
print()
print("═" * 110)
print("  대표 3정책 최종 KPI 종합 (Lv4): 균형/안정/효율")
print("═" * 110)
print(f"  {'KPI':<10} {'BAL':>10} {'STAB':>10} {'EFF':>10} {'Best':<6} {'Direction'}")
print("  " + "─" * 70)
kpis = [
    ("Reward",  "avg_reward",  "high"),
    ("VPR",     "avg_vpr",     "high"),
    ("WBI",     "avg_wbi",     "high"),
    ("OSR",     "avg_osr",     "low"),
    ("PSR",     "avg_psr",     "low"),
    ("CWVR",    "avg_cwvr",    "low"),
]
for name, key, direction in kpis:
    bl_v = bl_pm.get(key, 0) if bl_pm else float('nan')
    sf_v = sf_pm.get(key, 0)
    ef_v = ef_pm.get(key, 0)
    vals = {"BAL": bl_v, "STAB": sf_v, "EFF": ef_v}
    if direction == "high":
        best = max(vals, key=vals.get)
    else:
        best = min(vals, key=vals.get)
    arrow = "↑ better" if direction == "high" else "↓ better"
    print(f"  {name:<10} {bl_v:>10.4f} {sf_v:>10.4f} {ef_v:>10.4f}   {best:<6} {arrow}")
