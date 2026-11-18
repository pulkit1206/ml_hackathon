"""Model comparison table + plots, generated only from results/models/experiments.csv and preds/.

    python scripts/plot_comparison.py                       # latest run of every model
    python scripts/plot_comparison.py LogisticRegression KNN RandomForest LightGBM MLP
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from common import MODELS, PLOTS

exp = pd.read_csv(MODELS / "experiments.csv")
latest = exp.sort_values("timestamp").groupby("model").tail(1).set_index("model")
if len(sys.argv) > 1:
    latest = latest.loc[sys.argv[1:]]
cols = ["family", "n_features", "cv_macro_f1_mean", "cv_top1_mean", "cv_top1_std", "cv_train_top1_mean",
        "test_top1", "test_accuracy", "test_precision", "test_recall", "test_macro_f1", "test_log_loss",
        "train_time_s", "inference_ms_per_1k_rows"]
tab = latest[cols].round(4)
tab.to_csv(MODELS / "comparison_table.csv")
print(tab.to_string())
names = list(latest.index); x = range(len(names)); w = 0.38


def bar(col, title, fname, err=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, latest[col], yerr=latest[err] if err else None, capsize=4, color="#3b6ea5")
    for i, v in enumerate(latest[col]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title); plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(PLOTS / f"{fname}.png", dpi=150); plt.close()


bar("cv_top1_mean", "Cross-validation Top-1 accuracy (mean ± std over GroupKFold folds)", "cv_top1", "cv_top1_std")
bar("cv_macro_f1_mean", "Cross-validation Macro F1 (mean ± std)", "cv_macro_f1", "cv_macro_f1_std")
bar("test_log_loss", "Student-test log loss (lower is better)", "test_log_loss")
bar("inference_ms_per_1k_rows", "Inference time (ms per 1,000 candidate rows)", "inference_time")

for (a, b, la, lb, fname, title) in (
        ("test_top1", "test_macro_f1", "Top-1 accuracy", "Macro F1", "test_top1_f1", "Student-test performance"),
        ("cv_train_top1_mean", "cv_top1_mean", "training folds", "validation folds", "train_vs_val_top1",
         "Top-1: training vs validation folds (over/under-fitting)")):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - w / 2 for i in x], latest[a], w, label=la, color="#9bb7d4")
    ax.bar([i + w / 2 for i in x], latest[b], w, label=lb, color="#3b6ea5")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20); ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right"); ax.set_title(title); plt.tight_layout()
    plt.savefig(PLOTS / f"{fname}.png", dpi=150); plt.close()

for n in names:
    f = MODELS / "preds" / f"{n}.parquet"
    if f.exists():
        p = pd.read_parquet(f)
        ConfusionMatrixDisplay(confusion_matrix(p.label, (p.proba >= 0.5).astype(int)),
                               display_labels=["not reference arc", "reference arc"]).plot(values_format="d", cmap="Blues")
        plt.title(f"{n}: student-test confusion matrix (threshold 0.5)"); plt.tight_layout()
        plt.savefig(PLOTS / f"confusion_{n}.png", dpi=150); plt.close()
print(f"\nplots in {PLOTS}")
