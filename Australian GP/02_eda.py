# -*- coding: utf-8 -*-
"""
F1 Australian GP Podium Predictor
===================================
Step 2: Exploratory Data Analysis (EDA)
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv("aus_gp_dataset.csv")

C_PODIUM = "#E8C132"
C_NO     = "#3A3A3A"
C_RED    = "#DC0000"
C_SILVER = "#9B9B9B"
C_BG     = "#1A1A2E"
C_TEXT   = "#FFFFFF"
C_GRID   = "#333355"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "text.color": C_TEXT, "grid.color": C_GRID, "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(20, 24), facecolor=C_BG)
fig.suptitle("F1 Australian GP  --  Podium Prediction EDA (2014-2026)",
             fontsize=22, fontweight="bold", color=C_TEXT, y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.35)

# 1. Podium Rate by Qualifying Position
ax1 = fig.add_subplot(gs[0, :2])
quali_podium = df[df["quali_position"] <= 10].groupby("quali_position")["podium"].mean() * 100
bars = ax1.bar(quali_podium.index, quali_podium.values,
               color=[C_PODIUM if i < 3 else C_SILVER if i < 6 else C_NO
                      for i in range(len(quali_podium))],
               edgecolor="none", width=0.7)
for bar, val in zip(bars, quali_podium.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=9, color=C_TEXT)
ax1.set_title("Podium Rate by Qualifying Position", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Qualifying Position")
ax1.set_ylabel("Podium %")
ax1.set_xticks(range(1, 11))
ax1.grid(axis="y")
ax1.set_ylim(0, 105)

# 2. Team podium counts
ax2 = fig.add_subplot(gs[0, 2])
team_pods = df[df["podium"]==1]["team"].value_counts().head(8)
colors_t = [C_PODIUM if t in ["Mercedes","Ferrari","Red Bull","McLaren"] else C_SILVER
            for t in team_pods.index]
ax2.barh(team_pods.index[::-1], team_pods.values[::-1], color=colors_t[::-1], edgecolor="none")
ax2.set_title("Podiums by Team\n(2014-2026)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Podium Finishes")
ax2.grid(axis="x")

# 3. Qualifying vs Race Finish scatter
ax3 = fig.add_subplot(gs[1, :2])
colors_s = [C_PODIUM if p else C_NO for p in df["podium"]]
ax3.scatter(df["quali_position"], df["finish_position"], c=colors_s, alpha=0.7, s=60, edgecolors="none")
ax3.plot([1, 20], [1, 20], "--", color=C_SILVER, alpha=0.4, linewidth=1)
ax3.set_xlim(0.5, 20.5)
ax3.set_ylim(0.5, 21)
ax3.invert_yaxis()
ax3.set_title("Qualifying Position vs Race Finish  (Yellow = Podium)", fontsize=13, fontweight="bold", pad=10)
ax3.set_xlabel("Qualifying Position")
ax3.set_ylabel("Race Finish Position")
ax3.grid()
ax3.legend(handles=[mpatches.Patch(color=C_PODIUM, label="Podium"),
                    mpatches.Patch(color=C_NO, label="No podium")], loc="lower right")

# 4. Feature correlation
ax4 = fig.add_subplot(gs[1, 2])
features = ["quali_position", "grid_position", "team_tier", "champ_pos_before",
            "team_pos_before", "aus_hist_avg_pos", "aus_hist_races"]
correlations = df[features + ["podium"]].corr()["podium"].drop("podium").sort_values()
ax4.barh(correlations.index, correlations.values,
         color=[C_RED if v < 0 else C_PODIUM for v in correlations.values], edgecolor="none")
ax4.axvline(0, color=C_TEXT, linewidth=0.8)
ax4.set_title("Feature Correlation\nwith Podium", fontsize=13, fontweight="bold", pad=10)
ax4.set_xlabel("Pearson r")
ax4.grid(axis="x")

# 5. Podium rate by AUS history
ax5 = fig.add_subplot(gs[2, 0])
bins = [0, 5, 10, 15, 20, 25]
labels = ["1-5", "6-10", "11-15", "16-20", "21+"]
df["hist_bin"] = pd.cut(df["aus_hist_avg_pos"].fillna(25), bins=bins, labels=labels, right=True)
hist_rate = df.groupby("hist_bin", observed=True)["podium"].mean() * 100
ax5.bar(hist_rate.index, hist_rate.values, color=C_PODIUM, edgecolor="none")
ax5.set_title("Podium % by Historical\nAvg Finish at AUS", fontsize=11, fontweight="bold", pad=8)
ax5.set_xlabel("Historical Avg Finish Bin")
ax5.set_ylabel("Podium %")
ax5.grid(axis="y")

# 6. Top podium drivers
ax6 = fig.add_subplot(gs[2, 1])
top_drivers = df[df["podium"]==1]["driver"].value_counts().head(8)
team_map = dict(zip(df["driver"], df["team"]))
team_colors = {"Mercedes": C_SILVER, "Ferrari": C_RED, "Red Bull": "#3671C6", "McLaren": "#FF8000"}
colors_d = [team_colors.get(team_map.get(d, ""), C_GRID) for d in top_drivers.index]
ax6.barh(top_drivers.index[::-1], top_drivers.values[::-1], color=colors_d[::-1], edgecolor="none")
ax6.set_title("Most Podiums\n(2014-2026)", fontsize=11, fontweight="bold", pad=8)
ax6.set_xlabel("Podium Count")
ax6.grid(axis="x")

# 7. Year-by-year qualifying positions of podium finishers
ax7 = fig.add_subplot(gs[2, 2])
pod_df = df[df["podium"] == 1][["year", "quali_position"]]
years_sorted = sorted(pod_df["year"].unique())
ax7.boxplot([pod_df[pod_df["year"] == y]["quali_position"].tolist() for y in years_sorted],
            positions=range(len(years_sorted)), patch_artist=True, widths=0.5,
            boxprops=dict(facecolor=C_PODIUM, alpha=0.6, color=C_TEXT),
            medianprops=dict(color=C_TEXT, linewidth=2),
            whiskerprops=dict(color=C_TEXT), capprops=dict(color=C_TEXT),
            flierprops=dict(marker="o", color=C_PODIUM, markersize=4))
ax7.set_xticks(range(len(years_sorted)))
ax7.set_xticklabels(years_sorted, rotation=45, fontsize=8)
ax7.set_title("Quali Pos of Podium\nFinishers per Year", fontsize=11, fontweight="bold", pad=8)
ax7.set_ylabel("Qualifying Position")
ax7.grid(axis="y")

# 8. Team tier podium breakdown
ax8 = fig.add_subplot(gs[3, 0])
tier_podium = df.groupby("team_tier")["podium"].agg(["sum", "count"])
tier_podium["rate"] = tier_podium["sum"] / tier_podium["count"] * 100
ax8.bar(["Tier 1\n(Top teams)", "Tier 2\n(Midfield)", "Tier 3\n(Backmarkers)"][:len(tier_podium)],
        tier_podium["rate"].values, color=[C_PODIUM, C_SILVER, C_NO], edgecolor="none")
ax8.set_title("Podium Rate by\nTeam Tier", fontsize=11, fontweight="bold", pad=8)
ax8.set_ylabel("Podium %")
ax8.grid(axis="y")

# 9. Race finish distribution for Quali P1/P2/P3
ax9 = fig.add_subplot(gs[3, 1:])
top3_qualis = df[df["quali_position"] <= 5]
finish_counts = top3_qualis.groupby(["quali_position", "finish_position"]).size().unstack(fill_value=0)
finish_pct = finish_counts.div(finish_counts.sum(axis=1), axis=0) * 100
positions_plot = [1, 2, 3]
x = np.arange(len(positions_plot))
width = 0.12
finish_labels = [1, 2, 3, 4, 5, 6]
for i, fp in enumerate(finish_labels):
    vals = [finish_pct.loc[qp, fp] if fp in finish_pct.columns and qp in finish_pct.index else 0
            for qp in positions_plot]
    ax9.bar(x + i * width, vals, width, label=f"P{fp}",
            color=C_PODIUM if fp <= 3 else C_SILVER if fp <= 5 else C_NO,
            alpha=0.9 if fp <= 3 else 0.5, edgecolor="none")
ax9.set_xticks(x + width * (len(finish_labels)/2 - 0.5))
ax9.set_xticklabels([f"Quali P{p}" for p in positions_plot])
ax9.set_title("Race Finish Distribution for Drivers Starting P1 / P2 / P3",
              fontsize=11, fontweight="bold", pad=8)
ax9.set_ylabel("% of races")
ax9.legend(title="Finish Pos", loc="upper right", fontsize=8,
           facecolor=C_BG, edgecolor=C_GRID, labelcolor=C_TEXT)
ax9.grid(axis="y")

os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
plt.savefig(os.path.join(BASE_DIR, "plots", "eda_overview.png"), dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()

print("[OK] EDA plots saved: plots/eda_overview.png")
print("\n-- Key Insights ---------------------------------------------")
print(f"Podium from top-3 qualifying : {df[df['quali_position']<=3]['podium'].mean():.1%}")
print(f"Podium from P4-P6 qualifying : {df[(df['quali_position']>=4)&(df['quali_position']<=6)]['podium'].mean():.1%}")
print(f"Podium from P7+  qualifying  : {df[df['quali_position']>=7]['podium'].mean():.1%}")
print(f"\nTier 1 team podium rate : {df[df['team_tier']==1]['podium'].mean():.1%}")
print(f"Tier 2 team podium rate : {df[df['team_tier']==2]['podium'].mean():.1%}")
print(f"Tier 3 team podium rate : {df[df['team_tier']==3]['podium'].mean():.1%}")
print(f"\nCorrelation  quali_pos vs podium : {df['quali_position'].corr(df['podium']):.3f}")