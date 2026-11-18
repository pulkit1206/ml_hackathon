# Initial experiment (superseded)

First 5-model comparison (11:30 checkpoint): 3-fold GroupKFold on 80 training trips, 10 features, 50 m radius.

Issue found afterwards: every trip was projected with one fixed projection (UTM zone 36N, EPSG:32636).
The EDA showed that trips are spread worldwide (about -100 to +137 degrees longitude), so for most trips
this distorted distances and especially directions (heading_diff). It also inflated the training candidate
miss rate (14%). The labels here came from a simple greedy forward alignment.

The final pipeline (pathsnap/, scripts/) fixes this with a per-trip azimuthal-equidistant projection and uses
the team's dynamic-programming labels (notebook 02). These files are kept for transparency only. They are
not used for the final results.
