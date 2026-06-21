# Figure 3: 최종 라운드 KPI 막대 그래프
fig3, ax = plt.subplots(figsize=(12, 6.5))
metrics_for_bar = ["VPR", "WBI", "1−OSR", "1−PSR", "1−CWVR"]
n_metrics = len(metrics_for_bar)
n_policies = len(all_results)                         # ★ v13: 3-way 자동 대응
x = np.arange(n_metrics)
bar_width = min(0.32, 0.85 / n_policies)              # 정책 수에 따라 자동 조정

def get_kpi(res, name):
    pm = res["per_round_metrics"][-1]
    return {
        "VPR":     pm["avg_vpr"],
        "WBI":     pm["avg_wbi"],
        "1−OSR":   1.0 - pm["avg_osr"],
        "1−PSR":   1.0 - pm["avg_psr"],
        "1−CWVR":  1.0 - pm.get("avg_cwvr", 0.0),
    }[name]

for i, (pn, res) in enumerate(all_results.items()):
    spec = POLICIES[pn]
    vals = [get_kpi(res, m) for m in metrics_for_bar]
    offset = (i - (n_policies - 1) / 2.0) * bar_width   # ★ n-bar 자동 정렬
    bars = ax.bar(x + offset, vals, bar_width,
                  color=spec["color"], label=pn, edgecolor="black", alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", fontsize=8 if n_policies >= 3 else 9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(metrics_for_bar, fontsize=11)
ax.set_ylabel("Score (higher is better)", fontweight="bold")
ax.set_title("Figure 3. Final-round KPIs (Lv4) — 6 policies",
             fontsize=13, fontweight="bold")
ax.set_ylim(0, 1.1)
ax.legend(loc="upper right", fontsize=11)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig3_path = f"{all_loggers[REF_POLICY].log_dir}/fig3_kpi_bar.png"
plt.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {fig3_path}")

# Figure 4: Radar
fig4 = plt.figure(figsize=(8.5, 8.5))
ax = fig4.add_subplot(111, projection="polar")
angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
angles += angles[:1]
for pn, res in all_results.items():
    spec = POLICIES[pn]
    vals = [get_kpi(res, m) for m in metrics_for_bar]
    vals += vals[:1]
    ax.plot(angles, vals, color=spec["color"], linewidth=2.5,
            marker=spec["marker"], markersize=9, label=pn)
    ax.fill(angles, vals, color=spec["color"], alpha=0.15)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_for_bar, fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_title("Figure 4. KPI Radar — 6 policies (higher is better)",
             fontsize=13, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=11)
ax.grid(True)
plt.tight_layout()
fig4_path = f"{all_loggers[REF_POLICY].log_dir}/fig4_radar.png"
plt.savefig(fig4_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"💾 Saved: {fig4_path}")
