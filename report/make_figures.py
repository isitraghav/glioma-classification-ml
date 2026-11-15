"""Bar charts for the report/slides, in one clean style: light-grey vs steel-blue bars, boxed axes,
values printed on the bars. Numbers are copied from the saved outputs of glioma_analysis.ipynb
(the gene importances come from the re-fitted Random Forest in app/artifacts/models.joblib)."""
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Cambria", "Times New Roman", "DejaVu Serif"], "font.size": 10,
    "axes.edgecolor": "#333333", "axes.linewidth": 0.8, "axes.labelcolor": "#111111",
    "xtick.color": "#111111", "ytick.color": "#111111", "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.titleweight": "bold", "axes.titlesize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "legend.frameon": False, "legend.fontsize": 9,
})
GREY, MID, BLUE, EDGE = "#c8c8c8", "#8c8c8c", "#3f6fa8", "#2b2b2b"
BAR = dict(edgecolor=EDGE, linewidth=0.6)
OUT = "report/figs/"

models = ["Logistic\nRegression", "Random\nForest", "XGBoost", "Decision\nTree", "KNN"]
cv   = [98.3, 98.3, 98.9, 93.1, 95.3]
val  = [98.6, 100.0, 98.6, 92.8, 91.5]
acc  = [98.5, 96.9, 96.9, 94.7, 93.1]
f1   = [98.4, 97.1, 97.0, 94.9, 93.4]
lo   = [95.7, 93.9, 93.8, 90.8, 89.2]
hi   = [100.0, 99.4, 99.3, 98.4, 97.1]


def label(ax, bars, vals, dy=0.25, fmt="{:.1f}"):
    """Value on top of each bar, rotated like the reference charts."""
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v),
                ha="center", va="bottom", rotation=90, fontsize=8.5)


def tidy(ax):
    ax.tick_params(axis="x", length=0)
    ax.set_axisbelow(True)


# 1. Test accuracy vs test macro-F1 (with bootstrap 95% CI on macro-F1)
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w, x = 0.32, range(len(models))
b1 = ax.bar([i - w / 2 for i in x], acc, w, color=GREY, label="Accuracy", **BAR)
b2 = ax.bar([i + w / 2 for i in x], f1, w, color=BLUE, label="Macro-F1 (95% CI)", **BAR)
ax.errorbar([i + w / 2 for i in x], f1, yerr=[[f - l for f, l in zip(f1, lo)], [h - f for f, h in zip(f1, hi)]],
            fmt="none", ecolor=EDGE, elinewidth=0.8, capsize=2.5)
label(ax, b1, acc)
for b, v, h in zip(b2, f1, hi):
    ax.text(b.get_x() + b.get_width() / 2, h + 0.25, f"{v:.1f}", ha="center", va="bottom", rotation=90, fontsize=8.5)
ax.set_xticks(list(x), models); ax.set_ylim(85, 104); ax.set_yticks(range(85, 101, 5))
ax.set_ylabel("Score (%)"); ax.set_title("Test set (n = 131)")
ax.legend(loc="upper right", ncol=2); tidy(ax)
plt.tight_layout(); plt.savefig(OUT + "test_f1_ci.png", dpi=220); plt.close()

# 2. Macro-F1 in cross-validation, validation and test
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.26
groups = [("5-fold CV (train)", cv, GREY), ("Validation (n = 66)", val, MID), ("Test (n = 131)", f1, BLUE)]
for k, (lab, vals, col) in enumerate(groups):
    bars = ax.bar([i + (k - 1) * w for i in x], vals, w, color=col, label=lab, **BAR)
    label(ax, bars, vals)
ax.set_xticks(list(x), models); ax.set_ylim(85, 104); ax.set_yticks(range(85, 101, 5))
ax.set_ylabel("Macro-F1 (%)"); ax.set_title("Macro-F1: cross-validation vs validation vs test")
ax.legend(loc="upper right", ncol=3); tidy(ax)
plt.tight_layout(); plt.savefig(OUT + "cv_val_test.png", dpi=220); plt.close()

# 3. Class distribution
classes, counts = ["IDHmut-non-codel", "IDHwt", "IDHmut-codel"], [256, 229, 169]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
bars = ax.bar(classes, counts, 0.5, color=BLUE, **BAR)
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v} ({v / 654:.1%})", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, 300); ax.set_ylabel("Patients"); ax.set_title("Patients per class (n = 654)"); tidy(ax)
plt.tight_layout(); plt.savefig(OUT + "cell13.png", dpi=220); plt.close()

# 4. Random Forest top-10 genes
genes = ["RBP1", "KIAA0495", "PSMB2", "MIER1", "FBXO42", "CSDE1", "FBXO17", "TMEM167B", "FKBP9", "EMP3"]
imp = [0.1056, 0.0532, 0.0460, 0.0417, 0.0385, 0.0369, 0.0360, 0.0324, 0.0322, 0.0319]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
bars = ax.barh(genes[::-1], imp[::-1], 0.6, color=BLUE, **BAR)
for b, v in zip(bars, imp[::-1]):
    ax.text(v + 0.0015, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
ax.set_xlim(0, 0.122); ax.set_xlabel("Importance (mean decrease in impurity)")
ax.set_title("Random Forest: top 10 genes"); ax.tick_params(axis="y", length=0); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(OUT + "cell61.png", dpi=220); plt.close()
print("done")
