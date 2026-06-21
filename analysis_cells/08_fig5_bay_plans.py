# Figure 5: 정책별 최종 라운드 bay plan
for policy_name, res in all_results.items():
    print(f"\n  ── {policy_name} Final Bay Plan ──")
    final_lv = res["levels"][-1]
    fig = plot_bay_plan(res["trained_model"], final_lv,
                         {"rw": res["phase_info"][-1]["rw_snapshot"],
                          **{k: v for k, v in CONFIG.items() if k != "rw"}},
                         all_loggers[policy_name],
                         algo_name=f"PPO_{policy_name}")
    print(f"  💾 Saved {policy_name} bay plan")
