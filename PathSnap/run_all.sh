#!/usr/bin/env bash
# Full PathSnap pipeline. Run from the PathSnap/ folder with the data in data/train and data/student_test.
#   bash run_all.sh            (about 45-60 min on a laptop; Optuna is the longest step)
set -euo pipefail
echo "== 1/10 candidate dataset";          python scripts/build_dataset.py
echo "== 2/10 feature importance";         python scripts/feature_importance.py
echo "== 3/10 five models";                python scripts/train_models.py
echo "== 4/10 LightGBM loss curves";       python scripts/lgbm_curves.py
echo "== 5/10 Optuna (Bayesian) tuning";   python scripts/optuna_search.py 40
echo "== 6/10 tuned model + export";       python scripts/train_tuned.py
python scripts/lgbm_curves.py tuned
echo "== 7/10 comparison plots";           python scripts/plot_comparison.py
echo "== 8/10 HMM decoder tuning";         python scripts/tune_decoder.py 20
echo "== 9/10 classifier vs HMM";          python scripts/evaluate_hmm.py
echo "== 10/10 folium maps";               python scripts/make_maps.py
echo "Done. Results in results/models, results/hmm, results/maps; model in artifacts/."
