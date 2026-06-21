# 정책별 보상 항목 분석 (Lv4 평가, BASE 가중치 환경) — ★ v13: 3 정책
print("═" * 80)
print("  BL / SF / EF — Reward Components (R1~R15) at Lv4, evaluated with BASE rewards")
print("═" * 80)

# Lv4 의 phase 2 (BASE) config 으로 평가 → 두 정책의 final policy 차이만 측정
for pn, res in all_results.items():
    print(f"\n  ── {pn}: {POLICIES[pn]['kor']} ──")
    cfg = copy.deepcopy(CONFIG)
    cfg["rw"] = dict(CONFIG["rw"])   # BASE 가중치
    comp = print_reward_component_table(res, cfg, GLOBAL_SEED)

print()
print("  ※ 세 정책 모두 BASE 보상으로 평가 → 학습된 정책 차이만 측정")
