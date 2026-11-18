# PathSnap – Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy GPS observations to a road network. For every GPS point a classifier scores the
nearby road arcs, and an HMM decoded with the Viterbi algorithm turns those scores into a connected route.
The trained system runs behind a FastAPI backend (Laptop 1). A Streamlit UI (Laptop 2) sends it a trip id
and shows the predicted route and the accuracy computed on the server.

## Repository layout

```
PathSnap/
├── notebooks/            01 dataset exploration (EDA), 02 route-label generation
├── pathsnap/             core package
│   ├── io.py             readers for .track / .nodes / .arcs / .route
│   ├── geo.py            per-trip metric projection (azimuthal equidistant centred on the trip)
│   ├── labels.py         reference arc per GPS point (team labels; same monotonic alignment as notebook 02)
│   ├── features.py       candidate generation (arcs within 50 m) + 10 engineered features
│   ├── metrics.py        Top-1, macro F1, log loss, precision/recall, candidate miss rate
│   ├── graph.py          directed road graph, bounded shortest paths, route continuity
│   ├── hmm.py            HMM lattice, Viterbi decoding, classifier-only baseline
│   ├── pipeline.py       end-to-end prediction for one trip + server-side scoring
│   └── maps.py           Folium maps
├── scripts/              experiments (run from PathSnap/)
├── api/main.py           FastAPI backend (Laptop 1)
├── ui/web/index.html     map-style evaluator UI (Laptop 2); ui/web/console.html = dark console; ui/app.py = Streamlit
├── artifacts/            final model, feature list, best hyper-parameters, HMM parameters
├── results/
│   ├── eda/, labels/     outputs of the notebooks
│   ├── models/           experiments.csv, comparison table, feature importance, Optuna trials, plots
│   ├── hmm/              decoder tuning, classifier-only vs HMM on student-test, plots
│   └── maps/             Folium HTML maps
├── run_all.sh            runs the whole experimental pipeline
└── requirements.txt
```

## Setup

```bash
cd PathSnap
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Data (not committed – it is large): put the extracted trip folders here

```
PathSnap/data/train/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
PathSnap/data/student_test/<trip_id>/...
```

The evaluator partition is never used during development.

## Reproduce all results

```bash
bash run_all.sh
```

| Step | Script | Output |
|---|---|---|
| Candidate dataset | `scripts/build_dataset.py` | `results/models/dataset_summary.csv` (rows, miss rate per trip) |
| Feature selection | `scripts/feature_importance.py` | `feature_importance.csv/.png`, `selected_features.json` |
| 5-model comparison | `scripts/train_models.py` | `experiments.csv`, `experiments_folds.csv` |
| LightGBM loss curves | `scripts/lgbm_curves.py` | `plots/lgbm_loss_curve_*.png` |
| Bayesian optimisation | `scripts/optuna_search.py` | `optuna.db`, `optuna_trials.csv`, `optuna_summary.json`, plots |
| Tuned model + export | `scripts/train_tuned.py` | `tuned_vs_untuned.csv`, `artifacts/final_model.joblib` |
| Comparison plots | `scripts/plot_comparison.py` | `comparison_table.csv`, `plots/*.png` |
| HMM decoder tuning | `scripts/tune_decoder.py` | `results/hmm/decoder_tuning_*`, `artifacts/hmm_params.json` |
| Classifier vs HMM | `scripts/evaluate_hmm.py` | `results/hmm/student_test_clf_vs_hmm.csv`, plots |
| Maps | `scripts/make_maps.py` | `results/maps/student_test_<trip>.html` |

## Method summary

* **Projection.** The trips are spread across the world (about −100° to +137° longitude), so every trip gets
  its own azimuthal-equidistant projection centred on its GPS points. All distances are in metres.
* **Labels.** Each GPS point gets the route arc chosen by a monotonic dynamic-programming alignment to the
  ordered `.route` (notebook 02). Points more than 100 m from their reference arc are excluded as unreliable.
* **Candidates & features.** Arcs within 50 m of a GPS point. Features: perpendicular distance, distance rank,
  distance to the nearest candidate, number of candidates, heading difference (vehicle vs arc direction),
  speed, time gap, arc length, projection fraction, and connection to the previous point's nearest arc.
  No feature uses the route, so training and inference use identical code.
* **Models.** Logistic Regression, KNN, Random Forest, LightGBM and MLP, compared with 3-fold GroupKFold by
  trip on the 80 training trips and evaluated on the 10 student-test trips. Primary metric: observation-level
  Top-1 accuracy.
* **Bayesian optimisation.** Optuna TPE on LightGBM, objective = CV Top-1 on training trips only.
* **HMM/Viterbi.** States = candidate arcs. Emission = classifier probability. Transition =
  exp(−|driving distance − GPS distance| / β) (Newson & Krumm 2009). Impossible transitions and gaps over
  60 s restart the chain. β and the emission weight are tuned on out-of-fold probabilities of training trips.

## Live demo (two laptops on the same network)

**Laptop 1 – API** (holds the model and the evaluation trips, including their reference routes):

```bash
cd PathSnap && source venv/bin/activate
export PATHSNAP_EVAL_DIR=data/evaluator_test      # folder with the 10 evaluation trip folders
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Find its IP address (macOS: `ipconfig getifaddr en0`). Every request is logged in this terminal.

**Laptop 2 – UI** (web console, recommended):

```bash
cd PathSnap
python3 -m http.server 8501 --directory ui/web
```

Open `http://localhost:8501/?api=http://<laptop-1-ip>:8000` in a browser (or type the API address in the
sidebar and press Connect). Pick a trip and press **Run map-matching**. The map shows the GPS observations,
the predicted route, a start marker and a red end pin, with accuracy metrics and a request log in the side panel. `console.html` is an alternative dark layout.
Leaflet is bundled in `ui/web/vendor/`. Only the map tiles need internet.

Alternative Streamlit UI: `PATHSNAP_API=http://<laptop-1-ip>:8000 streamlit run ui/app.py`

The UI only sends the selected trip id to `POST /predict` and displays the response. Reference routes never
leave Laptop 1. They are read only after prediction, to compute accuracy.

## Notes

* All reported numbers come from the CSV files written by the scripts above.
* AI assistance: parts of the code were written with the help of an AI assistant (Claude). The team ran all
  experiments, chose the configurations and wrote the analysis.
