# Figure 2: 라운드별 누적 보상 (Phase 영역 음영)
n_rounds = CONFIG["total_rounds"]
fig, axes = plt.subplots(1, n_rounds, figsize=(5 * n_rounds, 5), sharey=False)
if n_rounds == 1: axes = [axes]

for rnd in range(n_rounds):
    ax = axes[rnd]
    lv = all_results[REF_POLICY]["levels"][rnd]

    # 이 라운드의 phase 표시 (모든 정책 phase 동일 → 기준 정책 사용)
    sf_phase_info = all_results[REF_POLICY]["phase_info"][rnd]
    phase_label = f"Phase {sf_phase_info['phase']} ({sf_phase_info['mode']})"
    bg_color = "#E2EFDA" if sf_phase_info["phase"] == 1 else "#DEEBF7"
    ax.set_facecolor(bg_color)

    for policy_name, res in all_results.items():
        spec = POLICIES[policy_name]
        cb = res["callbacks"][rnd]
        if not cb.ep_rewards: continue
        cumulative = np.cumsum(cb.ep_rewards)
        steps_in_round = cb.timesteps
        final_val = cumulative[-1]
        ax.plot(steps_in_round, cumulative, color=spec["color"], linewidth=1.8,
                label=f"{policy_name} ({final_val:+,.0f})", alpha=0.9)

    ax.set_title(f"Round {rnd+1} (Lv{lv})\n{phase_label}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Step (within round)")
    if rnd == 0:
        ax.set_ylabel("Cumulative Reward (within round)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

fig.suptitle("Figure 2. Per-round Cumulative Reward — 6 policies (Phase 영역 음영)",
             fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig_path = f"{all_loggers[REF_POLICY].log_dir}/fig2_per_round_cumulative.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {fig_path}")
