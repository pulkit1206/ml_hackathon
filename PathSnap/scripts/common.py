"""Shared paths/settings for the scripts. Run every script from the PathSnap/ folder."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = Path(os.environ.get("PATHSNAP_DATA", ROOT / "data"))
TRAIN_DIR, TEST_DIR = DATA / "train", DATA / "student_test"
RES = ROOT / "results"
MODELS = RES / "models"
PLOTS = MODELS / "plots"
HMM_RES = RES / "hmm"
MAPS = RES / "maps"
ART = ROOT / "artifacts"
TEAM_LABELS = [RES / "labels" / "train_gps_route_labels.parquet",
               RES / "labels" / "student_test_gps_route_labels.parquet"]
TRAIN_PQ = MODELS / "train_candidates.parquet"
TEST_PQ = MODELS / "student_test_candidates.parquet"
SEED = 42
N_SPLITS = 3

for d in (MODELS, PLOTS, HMM_RES, MAPS, ART):
    d.mkdir(parents=True, exist_ok=True)


def selected_features():
    from pathsnap.features import FEATURES_ALL
    f = MODELS / "selected_features.json"
    return json.load(open(f)) if f.exists() else FEATURES_ALL
