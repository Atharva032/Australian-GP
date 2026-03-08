"""
F1 Australian GP Podium Predictor
===================================
Step 3: Model Training + Evaluation + 2026 Prediction

Approach: Two-stage
  Stage 1: XGBoost binary classifier (podium or not) — per driver
  Stage 2: XGBoost ranker (rank:pairwise) — predicts exact P1/P2/P3 order
            among the likely podium candidates

Evaluation: Leave-One-Year-Out cross-validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb# -*- coding: utf-8 -*-
"""
F1 Australian GP Podium Predictor
===================================
Step 3: Model Training + Evaluation + 2026 Prediction

Approach: Two-stage
  Stage 1: XGBoost binary classifier (podium or not) per driver
  Stage 2: Rank by predicted probability to get P1/P2/P3 order

Evaluation: Leave-One-Year-Out cross-validation
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -- Style -------------------------------------------------------------
C_BG    = "#1A1A2E"
C_GOLD  = "#E8C132"
C_RED   = "#DC0000"
C_SILVER = "#9B9B9B"
C_TEXT  = "#FFFFFF"
C_GRID  = "#333355"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "text.color": C_TEXT, "grid.color": C_GRID, "grid.alpha": 0.4,
})

# -- Load data ---------------------------------------------------------
df = pd.read_csv("aus_gp_dataset.csv")

le_driver = LabelEncoder()
df["driver_enc"] = le_driver.fit_transform(df["driver"])

FEATURES = [
    "quali_position",
    "grid_position",
    "team_tier",
    "champ_pos_before",
    "team_pos_before",
    "aus_hist_avg_pos",
    "aus_hist_races",
    "driver_enc",
]

TARGET_CLASS  = "podium"
TARGET_FINISH = "finish_position"

# -- Leave-One-Year-Out CV ---------------------------------------------
years      = sorted(df["year"].unique())
test_years = [y for y in years if y >= 2017]
oof_records = []

clf_params = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

print("=" * 55)
print("  Leave-One-Year-Out CV")
print("=" * 55)

for test_year in test_years:
    train_df = df[df["year"] != test_year]
    test_df  = df[df["year"] == test_year]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_CLASS]
    X_test  = test_df[FEATURES]

    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    probs = clf.predict_proba(X_test)[:, 1]

    test_df = test_df.copy()
    test_df["podium_prob"] = probs
    test_df["pred_rank"]   = test_df["podium_prob"].rank(ascending=False).astype(int)
    test_df["pred_podium"] = (test_df["pred_rank"] <= 3).astype(int)

    predicted_top3 = test_df.nsmallest(3, "pred_rank")
    actual_top3    = test_df.nsmallest(3, "finish_position")["driver"].tolist()

    correct    = sum(d in actual_top3 for d in predicted_top3["driver"].tolist())
    p1_correct = predicted_top3.iloc[0]["driver"] == actual_top3[0]

    print(f"\n[{test_year}] Predicted: {predicted_top3['driver'].tolist()}")
    print(f"          Actual:    {actual_top3}")
    print(f"          Podium hits: {correct}/3 | P1 correct: {p1_correct}")

    for _, row in test_df.iterrows():
        oof_records.append({
            "year":          test_year,
            "driver":        row["driver"],
            "actual_podium": row["podium"],
            "pred_podium":   row["pred_podium"],
            "podium_prob":   row["podium_prob"],
            "actual_finish": row["finish_position"],
        })

oof_df = pd.DataFrame(oof_records)

# -- Metrics -----------------------------------------------------------
acc  = accuracy_score(oof_df["actual_podium"], oof_df["pred_podium"])
prec = precision_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
rec  = recall_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
f1   = f1_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
auc  = roc_auc_score(oof_df["actual_podium"], oof_df["podium_prob"])

hit_rates = []
for yr in test_years:
    yr_df    = oof_df[oof_df["year"] == yr]
    actual   = set(yr_df[yr_df["actual_podium"] == 1]["driver"])
    predicted = set(yr_df.sort_values("podium_prob", ascending=False).head(3)["driver"])
    hit_rates.append(len(actual & predicted))

print(f"\n{'='*55}")
print("  Overall CV Metrics")
print(f"{'='*55}")
print(f"  Accuracy    : {acc:.3f}")
print(f"  Precision   : {prec:.3f}")
print(f"  Recall      : {rec:.3f}")
print(f"  F1 Score    : {f1:.3f}")
print(f"  ROC-AUC     : {auc:.3f}")
print(f"  Avg Podium Hits (out of 3): {np.mean(hit_rates):.2f}")
print(f"  Years with 3/3 correct    : {sum(h==3 for h in hit_rates)}/{len(hit_rates)}")
print(f"  Years with >=2/3 correct  : {sum(h>=2 for h in hit_rates)}/{len(hit_rates)}")

# -- Final model (train on all except 2026) ----------------------------
train_full = df[df["year"] < 2026]
final_clf  = xgb.XGBClassifier(**clf_params)
final_clf.fit(train_full[FEATURES], train_full[TARGET_CLASS], verbose=False)

# -- Predict 2026 AUS GP -----------------------------------------------
df_2026 = df[df["year"] == 2026].copy()
df_2026["podium_prob"] = final_clf.predict_proba(df_2026[FEATURES])[:, 1]
df_2026 = df_2026.sort_values("podium_prob", ascending=False).reset_index(drop=True)
df_2026["pred_rank"] = range(1, len(df_2026) + 1)

print(f"\n{'='*55}")
print("  2026 AUS GP Predictions vs Actual")
print(f"{'='*55}")
print(f"  {'Rank':<5} {'Driver':<22} {'Team':<16} {'Q':<4} {'Prob':>6}  Actual")
print("  " + "-" * 60)
for _, row in df_2026.head(8).iterrows():
    trophy = "[P]" if row["finish_position"] <= 3 else "   "
    print(f"  P{int(row['pred_rank'])}   {row['driver']:<22} {row['team']:<16} "
          f"Q{int(row['quali_position'])}   {row['podium_prob']:.1%}  "
          f"{trophy} Actual P{int(row['finish_position'])}")

# -- Plots -------------------------------------------------------------
fig = plt.figure(figsize=(20, 18), facecolor=C_BG)
fig.suptitle("F1 Australian GP  --  Model Results & 2026 Prediction",
             fontsize=20, fontweight="bold", color=C_TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# A. 2026 predicted podium probability
ax_a = fig.add_subplot(gs[0, :2])
top_n  = df_2026.head(10)
colors = [C_GOLD if row["finish_position"] <= 3 else C_SILVER if row["pred_rank"] <= 3 else "#555577"
          for _, row in top_n.iterrows()]
bars = ax_a.barh(top_n["driver"][::-1], top_n["podium_prob"][::-1] * 100,
                 color=colors[::-1], edgecolor="none")
ax_a.set_title("2026 AUS GP -- Predicted Podium Probability\n"
               "(Gold = correctly predicted & on actual podium  |  Silver = predicted but missed)",
               fontsize=11, fontweight="bold", pad=10)
ax_a.set_xlabel("Podium Probability %")
ax_a.axvline(50, color=C_GRID, linestyle="--", alpha=0.5)
for bar, (_, row) in zip(reversed(bars), top_n.iterrows()):
    ax_a.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
              f"  Q{int(row['quali_position'])} -> P{int(row['finish_position'])}",
              va="center", fontsize=8.5, color=C_TEXT)
ax_a.grid(axis="x")
ax_a.set_xlim(0, 115)

# B. Metrics summary
ax_b = fig.add_subplot(gs[0, 2])
metrics       = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc}
metric_colors = [C_GOLD, C_SILVER, C_SILVER, "#FF8000", "#44AA88"]
bars_m = ax_b.bar(metrics.keys(), metrics.values(), color=metric_colors, edgecolor="none")
for bar, v in zip(bars_m, metrics.values()):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_b.set_title("CV Metrics\n(LOO-Year)", fontsize=13, fontweight="bold", pad=10)
ax_b.set_ylim(0, 1.15)
ax_b.grid(axis="y")
ax_b.tick_params(axis="x", rotation=30)

# C. Feature importance
ax_c = fig.add_subplot(gs[1, 0])
fi = pd.Series(final_clf.feature_importances_, index=FEATURES).sort_values()
ax_c.barh(fi.index, fi.values,
          color=[C_GOLD if f in ["quali_position", "team_tier"] else C_SILVER for f in fi.index],
          edgecolor="none")
ax_c.set_title("Feature\nImportance", fontsize=11, fontweight="bold", pad=8)
ax_c.set_xlabel("Importance")
ax_c.grid(axis="x")

# D. Podium hit rate per year
ax_d = fig.add_subplot(gs[1, 1])
ax_d.bar(test_years, hit_rates,
         color=[C_GOLD if h == 3 else C_SILVER if h == 2 else C_RED for h in hit_rates],
         edgecolor="none")
ax_d.axhline(np.mean(hit_rates), color=C_TEXT, linestyle="--", alpha=0.5,
             label=f"Mean: {np.mean(hit_rates):.1f}")
ax_d.set_title("Correct Podium Picks\nper Year (out of 3)", fontsize=11, fontweight="bold", pad=8)
ax_d.set_xlabel("Year")
ax_d.set_ylabel("Hits")
ax_d.set_ylim(0, 3.5)
ax_d.set_yticks([0, 1, 2, 3])
ax_d.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID)
ax_d.grid(axis="y")
ax_d.tick_params(axis="x", rotation=45)

# E. Confusion matrix
ax_e = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(oof_df["actual_podium"], oof_df["pred_podium"])
ax_e.imshow(cm, cmap="YlOrRd", aspect="auto")
ax_e.set_xticks([0, 1]); ax_e.set_yticks([0, 1])
ax_e.set_xticklabels(["No Podium", "Podium"])
ax_e.set_yticklabels(["No Podium", "Podium"])
ax_e.set_xlabel("Predicted"); ax_e.set_ylabel("Actual")
ax_e.set_title("Confusion Matrix\n(OOF)", fontsize=11, fontweight="bold", pad=8)
for i in range(2):
    for j in range(2):
        ax_e.text(j, i, str(cm[i, j]), ha="center", va="center",
                  fontsize=14, fontweight="bold", color=C_TEXT)

# F. Probability calibration
ax_f = fig.add_subplot(gs[2, :2])
oof_copy = oof_df.copy()
oof_copy["prob_bin"] = pd.cut(oof_copy["podium_prob"], bins=np.linspace(0, 1, 11))
cal = oof_copy.groupby("prob_bin", observed=True)["actual_podium"].mean()
bin_centers = [(b.left + b.right) / 2 for b in cal.index]
ax_f.plot([0, 1], [0, 1], "--", color=C_GRID, label="Perfect calibration")
ax_f.scatter(bin_centers, cal.values, color=C_GOLD, s=80, zorder=5)
ax_f.plot(bin_centers, cal.values, color=C_GOLD, linewidth=2, label="Model calibration")
ax_f.fill_between(bin_centers, cal.values, bin_centers, alpha=0.15, color=C_GOLD)
ax_f.set_title("Probability Calibration -- How well do predicted probabilities match actual rates?",
               fontsize=11, fontweight="bold", pad=8)
ax_f.set_xlabel("Predicted Probability")
ax_f.set_ylabel("Actual Podium Rate")
ax_f.legend(facecolor=C_BG, edgecolor=C_GRID)
ax_f.grid()
ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1)

# G. 2026 predicted vs actual top 3
ax_g = fig.add_subplot(gs[2, 2])
pred_top3        = df_2026.head(3)["driver"].tolist()
actual_top3_2026 = df_2026[df_2026["finish_position"] <= 3].sort_values("finish_position")["driver"].tolist()
all_drivers      = list(dict.fromkeys(pred_top3 + actual_top3_2026))
pred_set, actual_set = set(pred_top3), set(actual_top3_2026)

bar_info = []
for d in all_drivers:
    if d in pred_set and d in actual_set:
        c = C_GOLD
    elif d in pred_set:
        c = C_SILVER
    else:
        c = C_RED
    prob = df_2026[df_2026["driver"] == d]["podium_prob"].values[0]
    bar_info.append((d, prob * 100, c))

ax_g.barh([b[0] for b in bar_info], [b[1] for b in bar_info],
          color=[b[2] for b in bar_info], edgecolor="none")
ax_g.set_title("2026 Pred vs Actual\nTop 3", fontsize=11, fontweight="bold", pad=8)
ax_g.set_xlabel("Podium Prob %")
ax_g.grid(axis="x")
ax_g.legend(handles=[
    Patch(color=C_GOLD,   label="Predicted & Actual"),
    Patch(color=C_SILVER, label="Predicted only"),
    Patch(color=C_RED,    label="Actual only (missed)"),
], fontsize=7, facecolor=C_BG, edgecolor=C_GRID)

# -- Save outputs ------------------------------------------------------
os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"),  exist_ok=True)

plt.savefig(os.path.join(BASE_DIR, "plots", "model_results.png"),
            dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()

df_2026.to_csv(os.path.join(BASE_DIR, "data", "predictions_2026.csv"), index=False)

print("\n[OK] Model plots saved  : plots/model_results.png")
print("[OK] Predictions saved  : data/predictions_2026.csv")
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────
C_BG    = "#1A1A2E"
C_GOLD  = "#E8C132"
C_RED   = "#DC0000"
C_SILVER = "#9B9B9B"
C_TEXT  = "#FFFFFF"
C_GRID  = "#333355"

plt.rcParams.update({
    "figure.facecolor": C_BG, "axes.facecolor": C_BG,
    "axes.edgecolor": C_GRID, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "text.color": C_TEXT, "grid.color": C_GRID, "grid.alpha": 0.4,
})

# ── Load data ──────────────────────────────────────────────────────
df = pd.read_csv("aus_gp_dataset.csv")

# Encode drivers
le_driver = LabelEncoder()
df["driver_enc"] = le_driver.fit_transform(df["driver"])

FEATURES = [
    "quali_position",
    "grid_position",
    "team_tier",
    "champ_pos_before",
    "team_pos_before",
    "aus_hist_avg_pos",
    "aus_hist_races",
    "driver_enc",
]

TARGET_CLASS  = "podium"
TARGET_FINISH = "finish_position"

# ─────────────────────────────────────────────────────────────
# LEAVE-ONE-YEAR-OUT CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────

years = sorted(df["year"].unique())
# Use years 2016+ for test (need enough training data before them)
test_years = [y for y in years if y >= 2017]

oof_records = []
clf_params = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    verbosity=0,
)

print("=" * 55)
print("  Leave-One-Year-Out CV")
print("=" * 55)

for test_year in test_years:
    train_df = df[df["year"] != test_year]
    test_df  = df[df["year"] == test_year]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET_CLASS]
    X_test  = test_df[FEATURES]
    y_test  = test_df[TARGET_CLASS]

    clf = xgb.XGBClassifier(**clf_params)
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train)], verbose=False)

    probs  = clf.predict_proba(X_test)[:, 1]
    preds  = (probs >= 0.4).astype(int)

    # Rank by predicted probability → top 3 are predicted podium
    test_df = test_df.copy()
    test_df["podium_prob"] = probs
    test_df["pred_rank"]   = test_df["podium_prob"].rank(ascending=False).astype(int)
    test_df["pred_podium"] = (test_df["pred_rank"] <= 3).astype(int)

    # Predicted P1, P2, P3
    predicted_top3 = test_df.nsmallest(3, "pred_rank")[["driver", "pred_rank", "podium_prob", "finish_position", "podium"]]
    actual_top3    = test_df.nsmallest(3, "finish_position")["driver"].tolist()

    correct = sum([d in actual_top3 for d in predicted_top3["driver"].tolist()])
    p1_correct = predicted_top3.iloc[0]["driver"] == test_df.nsmallest(1, "finish_position")["driver"].iloc[0]

    print(f"\n[{test_year}] Predicted: {predicted_top3['driver'].tolist()}")
    print(f"          Actual:    {actual_top3}")
    print(f"          Podium hits: {correct}/3 | P1 correct: {p1_correct}")

    for _, row in test_df.iterrows():
        oof_records.append({
            "year": test_year,
            "driver": row["driver"],
            "actual_podium": row["podium"],
            "pred_podium": row["pred_podium"],
            "podium_prob": row["podium_prob"],
            "actual_finish": row["finish_position"],
        })

oof_df = pd.DataFrame(oof_records)

# ── Metrics ───────────────────────────────────────────────────────
acc  = accuracy_score(oof_df["actual_podium"], oof_df["pred_podium"])
prec = precision_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
rec  = recall_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
f1   = f1_score(oof_df["actual_podium"], oof_df["pred_podium"], zero_division=0)
auc  = roc_auc_score(oof_df["actual_podium"], oof_df["podium_prob"])

# Podium hit rate (how many of 3 actual podium drivers were in predicted top 3)
hit_rates = []
for yr in test_years:
    yr_df = oof_df[oof_df["year"] == yr]
    actual = set(yr_df[yr_df["actual_podium"]==1]["driver"])
    predicted = set(yr_df.nsmallest(3, "podium_prob")["driver"] if len(yr_df) > 0 else [])
    predicted = set(yr_df.sort_values("podium_prob", ascending=False).head(3)["driver"])
    hits = len(actual & predicted)
    hit_rates.append(hits)

print(f"\n{'='*55}")
print(f"  Overall CV Metrics")
print(f"{'='*55}")
print(f"  Accuracy    : {acc:.3f}")
print(f"  Precision   : {prec:.3f}")
print(f"  Recall      : {rec:.3f}")
print(f"  F1 Score    : {f1:.3f}")
print(f"  ROC-AUC     : {auc:.3f}")
print(f"  Avg Podium Hits (out of 3): {np.mean(hit_rates):.2f}")
print(f"  Years with 3/3 correct: {sum(h==3 for h in hit_rates)}/{len(hit_rates)}")
print(f"  Years with ≥2/3 correct: {sum(h>=2 for h in hit_rates)}/{len(hit_rates)}")

# ─────────────────────────────────────────────────────────────
# FINAL MODEL — Train on ALL data except 2026
# ─────────────────────────────────────────────────────────────

train_full = df[df["year"] < 2026]
X_full = train_full[FEATURES]
y_full = train_full[TARGET_CLASS]

final_clf = xgb.XGBClassifier(**clf_params)
final_clf.fit(X_full, y_full, verbose=False)

# ─────────────────────────────────────────────────────────────
# PREDICT 2026 AUSTRALIAN GP
# ─────────────────────────────────────────────────────────────

df_2026 = df[df["year"] == 2026].copy()
X_2026  = df_2026[FEATURES]
df_2026["podium_prob"] = final_clf.predict_proba(X_2026)[:, 1]
df_2026 = df_2026.sort_values("podium_prob", ascending=False).reset_index(drop=True)
df_2026["pred_rank"] = range(1, len(df_2026)+1)

print(f"\n{'='*55}")
print("  2026 AUS GP Predictions vs Actual")
print(f"{'='*55}")
print(f"  {'Rank':<5} {'Driver':<22} {'Team':<16} {'Q':<4} {'Prob':>6}  {'Actual'}")
print("  " + "-" * 60)
for _, row in df_2026.head(8).iterrows():
    marker = "🏆" if row["finish_position"] <= 3 else "  "
    print(f"  P{int(row['pred_rank'])}   {row['driver']:<22} {row['team']:<16} "
          f"Q{int(row['quali_position'])}   {row['podium_prob']:.1%}  "
          f"{marker} Actual: P{int(row['finish_position'])}")

# ─────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 18), facecolor=C_BG)
fig.suptitle("🏎  F1 Australian GP — Model Results & 2026 Prediction",
             fontsize=20, fontweight="bold", color=C_TEXT, y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35)

# ── A. 2026 Predicted Podium Probability ──────────────────────────
ax_a = fig.add_subplot(gs[0, :2])
top_n = df_2026.head(10)
colors = [C_GOLD if row["finish_position"] <= 3 else C_SILVER if row["pred_rank"] <= 3 else "#555577"
          for _, row in top_n.iterrows()]
bars = ax_a.barh(top_n["driver"][::-1], top_n["podium_prob"][::-1] * 100,
                 color=colors[::-1], edgecolor="none")
ax_a.set_title("2026 AUS GP — Predicted Podium Probability\n(🟡 = Correctly predicted & on actual podium  |  Grey top-3 = Predicted but not podium)",
               fontsize=11, fontweight="bold", pad=10)
ax_a.set_xlabel("Podium Probability %")
ax_a.axvline(50, color=C_GRID, linestyle="--", alpha=0.5)
for bar, (_, row) in zip(reversed(bars), top_n.iterrows()):
    ax_a.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
              f"  Q{int(row['quali_position'])} → P{int(row['finish_position'])}",
              va="center", fontsize=8.5, color=C_TEXT)
ax_a.grid(axis="x")
ax_a.set_xlim(0, 115)

# ── B. Metrics summary ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])
metrics = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC-AUC": auc}
metric_colors = [C_GOLD, C_SILVER, C_SILVER, "#FF8000", "#44AA88"]
bars_m = ax_b.bar(metrics.keys(), metrics.values(), color=metric_colors, edgecolor="none")
for bar, v in zip(bars_m, metrics.values()):
    ax_b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
              f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax_b.set_title("CV Metrics\n(LOO-Year)", fontsize=13, fontweight="bold", pad=10)
ax_b.set_ylim(0, 1.15)
ax_b.grid(axis="y")
ax_b.tick_params(axis="x", rotation=30)

# ── C. Feature Importance ──────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
fi = pd.Series(final_clf.feature_importances_, index=FEATURES).sort_values()
fi_colors = [C_GOLD if f in ["quali_position", "team_tier"] else C_SILVER for f in fi.index]
ax_c.barh(fi.index, fi.values, color=fi_colors, edgecolor="none")
ax_c.set_title("Feature\nImportance", fontsize=11, fontweight="bold", pad=8)
ax_c.set_xlabel("Importance")
ax_c.grid(axis="x")

# ── D. Podium Hit Rate per year ────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
bar_colors_y = [C_GOLD if h == 3 else C_SILVER if h == 2 else C_RED for h in hit_rates]
ax_d.bar(test_years, hit_rates, color=bar_colors_y, edgecolor="none")
ax_d.axhline(np.mean(hit_rates), color=C_TEXT, linestyle="--", alpha=0.5,
             label=f"Mean: {np.mean(hit_rates):.1f}")
ax_d.set_title("Correct Podium Picks\nper Year (out of 3)", fontsize=11, fontweight="bold", pad=8)
ax_d.set_xlabel("Year")
ax_d.set_ylabel("Hits")
ax_d.set_ylim(0, 3.5)
ax_d.set_yticks([0, 1, 2, 3])
ax_d.legend(fontsize=8, facecolor=C_BG, edgecolor=C_GRID)
ax_d.grid(axis="y")
ax_d.tick_params(axis="x", rotation=45)

# ── E. Confusion matrix ────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(oof_df["actual_podium"], oof_df["pred_podium"])
im = ax_e.imshow(cm, cmap="YlOrRd", aspect="auto")
ax_e.set_xticks([0, 1]); ax_e.set_yticks([0, 1])
ax_e.set_xticklabels(["No Podium", "Podium"])
ax_e.set_yticklabels(["No Podium", "Podium"])
ax_e.set_xlabel("Predicted"); ax_e.set_ylabel("Actual")
ax_e.set_title("Confusion Matrix\n(OOF)", fontsize=11, fontweight="bold", pad=8)
for i in range(2):
    for j in range(2):
        ax_e.text(j, i, str(cm[i, j]), ha="center", va="center",
                  fontsize=14, fontweight="bold", color=C_TEXT)

# ── F. Probability calibration (predicted prob vs actual podium rate) ──
ax_f = fig.add_subplot(gs[2, :2])
oof_df_sorted = oof_df.copy()
bins = np.linspace(0, 1, 11)
oof_df_sorted["prob_bin"] = pd.cut(oof_df_sorted["podium_prob"], bins=bins)
cal = oof_df_sorted.groupby("prob_bin", observed=True)["actual_podium"].mean()
bin_centers = [(b.left + b.right) / 2 for b in cal.index]
ax_f.plot([0, 1], [0, 1], "--", color=C_GRID, label="Perfect calibration")
ax_f.scatter(bin_centers, cal.values, color=C_GOLD, s=80, zorder=5)
ax_f.plot(bin_centers, cal.values, color=C_GOLD, linewidth=2, label="Model calibration")
ax_f.fill_between(bin_centers, cal.values, bin_centers,
                  alpha=0.15, color=C_GOLD)
ax_f.set_title("Probability Calibration — How well do predicted probabilities match actual rates?",
               fontsize=11, fontweight="bold", pad=8)
ax_f.set_xlabel("Predicted Probability")
ax_f.set_ylabel("Actual Podium Rate")
ax_f.legend(facecolor=C_BG, edgecolor=C_GRID)
ax_f.grid()
ax_f.set_xlim(0, 1); ax_f.set_ylim(0, 1)

# ── G. 2026 Predicted vs Actual podium ──────────────────────────
ax_g = fig.add_subplot(gs[2, 2])
pred_top3 = df_2026.head(3)["driver"].tolist()
actual_top3_2026 = df_2026[df_2026["finish_position"] <= 3].sort_values("finish_position")["driver"].tolist()
all_drivers = list(dict.fromkeys(pred_top3 + actual_top3_2026))
pred_set = set(pred_top3)
actual_set = set(actual_top3_2026)

bar_info = []
for d in all_drivers:
    in_pred = d in pred_set
    in_actual = d in actual_set
    if in_pred and in_actual:
        c, label = C_GOLD, "✅ Both"
    elif in_pred:
        c, label = C_SILVER, "Predicted only"
    else:
        c, label = C_RED, "Actual only"
    prob = df_2026[df_2026["driver"]==d]["podium_prob"].values[0]
    bar_info.append((d, prob, c, label))

drivers_plot = [b[0] for b in bar_info]
probs_plot   = [b[1]*100 for b in bar_info]
colors_plot  = [b[2] for b in bar_info]
ax_g.barh(drivers_plot, probs_plot, color=colors_plot, edgecolor="none")
ax_g.set_title("2026 Pred vs Actual\nTop 3", fontsize=11, fontweight="bold", pad=8)
ax_g.set_xlabel("Podium Prob %")
ax_g.grid(axis="x")

from matplotlib.patches import Patch
legend_elements = [Patch(color=C_GOLD, label="Predicted & Actual"),
                   Patch(color=C_SILVER, label="Predicted only"),
                   Patch(color=C_RED, label="Actual only (missed)")]
ax_g.legend(handles=legend_elements, fontsize=7, facecolor=C_BG, edgecolor=C_GRID)

plt.savefig("plots/model_results.png",
            dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()

# Save final predictions
df_2026.to_csv("predictions_2026.csv", index=False)
print(f"\n✅ Model plots saved → plots/model_results.png")
print(f"✅ Predictions saved → data/predictions_2026.csv")
