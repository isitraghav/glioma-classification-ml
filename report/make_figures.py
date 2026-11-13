"""Summary charts for the report/slides. All numbers are copied from the saved outputs of glioma_analysis.ipynb."""
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
INK, INK2, BLUE, ORANGE, AQUA = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"

models = ["Logistic Regression", "Random Forest", "XGBoost", "Decision Tree", "KNN"]
cv   = [0.983, 0.983, 0.989, 0.931, 0.953]
val  = [0.986, 1.000, 0.986, 0.928, 0.915]
test = [0.984, 0.971, 0.970, 0.949, 0.934]
lo   = [0.957, 0.939, 0.938, 0.908, 0.892]
hi   = [1.000, 0.994, 0.993, 0.984, 0.971]

# 1. Test macro-F1 with bootstrap 95% CI (single series -> one hue, no legend)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ypos = list(range(len(models)))[::-1]
for y, m, t, l, h in zip(ypos, models, test, lo, hi):
    ax.plot([l, h], [y, y], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(t, y, "o", ms=9, color=BLUE, mec="#fcfcfb", mew=2)
    ax.text(h + 0.004, y, f"{t:.3f}  [{l:.3f}–{h:.3f}]", va="center", fontsize=9, color=INK2)
ax.set_yticks(ypos, models, color=INK)
ax.set_xlim(0.88, 1.035); ax.set_xlabel("Test macro-F1 (dot) with bootstrap 95% CI (line), n = 131")
ax.grid(axis="x", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("Held-out test performance of the five tuned models", loc="left", color=INK, fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig("report/figs/test_f1_ci.png", dpi=200); plt.close()

# 2. CV vs validation vs test macro-F1 (3 series -> fixed categorical order, shapes + legend)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
series = [("5-fold CV (train)", cv, BLUE, "o"), ("Validation (n = 66)", val, ORANGE, "s"), ("Test (n = 131)", test, AQUA, "D")]
for k, (lab, vals, col, mk) in enumerate(series):
    xs = [i + (k - 1) * 0.22 for i in range(len(models))]
    ax.plot(xs, vals, mk, ms=8, color=col, mec="#fcfcfb", mew=1.5, label=lab, ls="none")
ax.set_xticks(range(len(models)), [m.replace(" ", "\n", 1) if m == "Logistic Regression" else m for m in models])
ax.set_ylim(0.90, 1.005); ax.set_ylabel("Macro-F1")
ax.grid(axis="y", color="#e6e5e0", lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower left", fontsize=9)
ax.set_title("Macro-F1 across cross-validation, validation and test", loc="left", color=INK, fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig("report/figs/cv_val_test.png", dpi=200); plt.close()
print("done")
