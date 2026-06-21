# Figure 1: 누적 보상 학습 곡선 (X=Step, Y=Cumulative Reward)
fig, ax = plt.subplots(figsize=(13, 7))

phase_boundaries = {}      # 정책별 Phase 1→2 step 위치
final_cum_rewards = {}     # 정책별 최종 누적 보상

for policy_name, res in all_results.items():
    spec = POLICIES[policy_name]

    # 모든 라운드의 (step, episode_reward) 데이터 추출
    # cb.timesteps 는 reset_num_timesteps=False 로 글로벌 step 카운트
    all_steps   = []
    all_rewards = []
    for cb in res["callbacks"]:
        all_steps.extend(cb.timesteps)
        all_rewards.extend(cb.ep_rewards)

    if not all_steps:
        print(f"  ⚠ {policy_name}: callback data 비어있음")
        continue

    cumulative = np.cumsum(all_rewards)
    final_cum_rewards[policy_name] = cumulative[-1]

    ax.plot(all_steps, cumulative, color=spec["color"], linewidth=2.0, alpha=0.9,
            label=f"{policy_name}: {spec['kor']} (final = {cumulative[-1]:+,.0f})")

    # Phase 1→2 boundary (★ v13: BL은 phase 구분 없으므로 미표시)
    if not spec.get("is_baseline", False):
        n_phase1_rounds = len(spec["phase1_rounds"])
        if 0 < n_phase1_rounds <= len(res["callbacks"]):
            last_p1_cb = res["callbacks"][n_phase1_rounds - 1]
            if last_p1_cb.timesteps:
                boundary_step = last_p1_cb.timesteps[-1]
                phase_boundaries[policy_name] = boundary_step
                ax.axvline(boundary_step, color=spec["color"], linestyle=":",
                           linewidth=1.8, alpha=0.7)
                # Phase 라벨
                ymin, ymax = ax.get_ylim() if ax.get_ylim()[1] != 1 else (-1000, 1000)
                ax.annotate(f"{policy_name}\nP1→P2",
                            xy=(boundary_step, 0), xytext=(boundary_step, 0),
                            fontsize=9, fontweight="bold", color=spec["color"],
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3",
                                      facecolor="white", edgecolor=spec["color"], alpha=0.9))

ax.set_xlabel("Step (Global, across all curriculum rounds)",
              fontsize=13, fontweight="bold")
ax.set_ylabel("Cumulative Reward", fontsize=13, fontweight="bold")
ax.set_title("Figure 1. Cumulative Reward over Steps — 6 policies",
             fontsize=14, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(loc="best", fontsize=11, framealpha=0.95)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()

fig_path = f"{all_loggers[REF_POLICY].log_dir}/fig1_cumulative_reward_steps.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {fig_path}")
print()
print("  ── 최종 누적 보상 비교 (3-way) ──")
for pn, fr in final_cum_rewards.items():
    print(f"    {pn}: {fr:+,.0f}")
if final_cum_rewards:
    best = max(final_cum_rewards, key=final_cum_rewards.get)
    print(f"  → 최고: {best} ({final_cum_rewards[best]:+,.0f})")
