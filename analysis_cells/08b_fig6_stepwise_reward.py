# Figure 6: 스텝별 보상 학습 곡선 (per-episode reward, smoothed)
# - X = Global Step (모든 라운드 누적)
# - Y = Episode Reward (smoothed rolling mean)
# - Phase 1→2 전환점 + Curriculum 레벨 표시
fig, ax = plt.subplots(figsize=(14, 7.5))

y_min_all, y_max_all = float('inf'), float('-inf')

for policy_name, res in all_results.items():
    spec = POLICIES[policy_name]

    # 모든 라운드의 (step, episode_reward) 결합
    all_steps, all_rewards = [], []
    for cb in res["callbacks"]:
        all_steps.extend(cb.timesteps)
        all_rewards.extend(cb.ep_rewards)

    if not all_steps:
        print(f"  ⚠ {policy_name}: callback data 비어있음")
        continue

    steps_arr = np.array(all_steps)
    rew_arr   = np.array(all_rewards)

    # Raw episode rewards (옅은 색, 분산 확인용)
    ax.plot(steps_arr, rew_arr, color=spec["color"],
            linewidth=0.4, alpha=0.18)

    # Smoothed rolling mean (진한 색, 학습 트렌드)
    n_eps = len(rew_arr)
    window = max(50, n_eps // 100)   # adaptive window (~1% of episodes, ≥50)
    if n_eps >= window:
        kernel = np.ones(window) / window
        smoothed = np.convolve(rew_arr, kernel, mode='valid')
        smoothed_x = steps_arr[window-1:]
        ax.plot(smoothed_x, smoothed, color=spec["color"],
                linewidth=2.4, alpha=0.95,
                label=f"{policy_name}: {spec['kor']}  (window={window} eps)")
        y_min_all = min(y_min_all, float(smoothed.min()))
        y_max_all = max(y_max_all, float(smoothed.max()))

    # Phase 1→2 전환 표시 (★ v13: BL은 phase 구분 없으므로 미표시)
    if not spec.get("is_baseline", False):
        n_phase1 = len(spec["phase1_rounds"])
        if 0 < n_phase1 <= len(res["callbacks"]):
            last_p1_cb = res["callbacks"][n_phase1 - 1]
            if last_p1_cb.timesteps:
                boundary = last_p1_cb.timesteps[-1]
                ax.axvline(boundary, color=spec["color"], linestyle=":",
                           linewidth=1.8, alpha=0.7)
                ax.annotate(f"{policy_name}\nP1→P2",
                            xy=(boundary, 0), xytext=(boundary, 0),
                            fontsize=9, fontweight="bold", color=spec["color"],
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=spec["color"], alpha=0.9))

# Curriculum 레벨 박스 (상단)
if y_max_all != float('-inf'):
    y_top = y_max_all + (y_max_all - y_min_all) * 0.08
    sf_res = all_results.get(REF_POLICY)
    if sf_res:
        cum_start = 0
        for rnd, cb in enumerate(sf_res["callbacks"]):
            if not cb.timesteps: continue
            end_step = cb.timesteps[-1]
            mid = (cum_start + end_step) / 2.0
            ax.text(mid, y_top, f"Lv{sf_res['levels'][rnd]}",
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color='#444444', alpha=0.85,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="#F2F2F2", edgecolor='gray', alpha=0.9))
            cum_start = end_step

ax.set_xlabel("Step (Global, across all curriculum rounds)",
              fontsize=13, fontweight="bold")
ax.set_ylabel("Episode Reward (raw light / smoothed bold)",
              fontsize=13, fontweight="bold")
ax.set_title("Figure 6. Step-wise (per-episode) Reward Learning Curve — 6 policies",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()

fig_path = f"{all_loggers[REF_POLICY].log_dir}/fig6_stepwise_reward.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {fig_path}")

# ── 학습 안정성 진단 출력 (★ v13: BL 제외 — phase 전환이 없음) ──
print()
print("  ── 학습 곡선 진단 (Phase 2 안정성, baseline 제외) ──")
for pn, res in all_results.items():
    spec = POLICIES[pn]
    if spec.get("is_baseline", False):
        print(f"    {pn}: (baseline — phase 전환 없음, skip)")
        continue
    if not res["callbacks"]: continue
    n_p1 = len(spec["phase1_rounds"])
    if n_p1 == 0 or n_p1 >= len(res["callbacks"]): continue
    last_p1_rewards  = res["callbacks"][n_p1 - 1].ep_rewards
    first_p2_rewards = res["callbacks"][n_p1].ep_rewards
    if not last_p1_rewards or not first_p2_rewards: continue
    tail_p1  = float(np.mean(last_p1_rewards[-100:]))
    head_p2  = float(np.mean(first_p2_rewards[:100]))
    shock    = abs(head_p2 - tail_p1) / (abs(tail_p1) + 1e-6)
    print(f"    {pn}: Phase1 tail = {tail_p1:+7.1f}  →  Phase2 head = {head_p2:+7.1f}  "
          f"|  ReShock = {shock:.2f} {'(stable ✓)' if shock < 0.20 else '(check)'}")