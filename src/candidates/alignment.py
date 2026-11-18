from pathlib import Path
import numpy as np
import pandas as pd

from src.candidates.generator import generate_candidates, load_track, project_track
from src.candidates.geometry import load_road_network
from src.candidates.features import compute_track_kinematics


def load_route(route_path):
    """
    Load reference route arc IDs from a .route file in sequence order.
    """
    route_raw = np.loadtxt(route_path, dtype=int)
    route_raw = np.atleast_1d(route_raw).astype("int64")
    return route_raw


def validate_route_connectivity(network, route_arcs):
    """
    Validate that consecutive arcs in .route form a directed path
    (to_node_id[i] == from_node_id[i + 1]).
    """
    arcs_df = network["arcs"]
    if len(route_arcs) == 0:
        return {
            "route_length": 0,
            "unique_route_arcs": 0,
            "valid_arc_ids": True,
            "directed_breaks": 0,
            "is_continuous_directed_path": True,
        }

    valid_ids = bool(np.all((route_arcs >= 0) & (route_arcs < len(arcs_df))))
    if not valid_ids:
        return {
            "route_length": int(len(route_arcs)),
            "unique_route_arcs": int(len(np.unique(route_arcs))),
            "valid_arc_ids": False,
            "directed_breaks": -1,
            "is_continuous_directed_path": False,
        }

    r_arcs = arcs_df.iloc[route_arcs]
    if len(route_arcs) > 1:
        to_nodes = r_arcs["to_node_id"].to_numpy(dtype="int64")[:-1]
        from_nodes = r_arcs["from_node_id"].to_numpy(dtype="int64")[1:]
        breaks = int(np.sum(to_nodes != from_nodes))
    else:
        breaks = 0

    return {
        "route_length": int(len(route_arcs)),
        "unique_route_arcs": int(len(np.unique(route_arcs))),
        "valid_arc_ids": True,
        "directed_breaks": breaks,
        "is_continuous_directed_path": breaks == 0,
    }


def analyze_track_kinematics(network, track_df, unrealistic_speed_mps=50.0, jitter_dist_m=1.0):
    """
    Inspect timestamp units, actual dt, unrealistic speeds, and stationary GPS jitter
    without deleting or modifying any observations.
    """
    projected = project_track(track_df, network["crs"])
    kin = compute_track_kinematics(projected)

    t = track_df["timestamp"].to_numpy(dtype="float64")
    dt = np.diff(t) if len(t) > 1 else np.array([], dtype="float64")
    dx = np.diff(projected["x"].to_numpy(dtype="float64")) if len(projected) > 1 else np.array([], dtype="float64")
    dy = np.diff(projected["y"].to_numpy(dtype="float64")) if len(projected) > 1 else np.array([], dtype="float64")
    step_dist = np.hypot(dx, dy)

    speeds = kin["speed_mps"].dropna().to_numpy(dtype="float64")
    unrealistic_count = int(np.sum(speeds > unrealistic_speed_mps))
    zero_move_count = int(np.sum(step_dist == 0.0))
    jitter_count = int(np.sum((step_dist > 0.0) & (step_dist < jitter_dist_m)))

    return {
        "total_observations": int(len(track_df)),
        "dt_min_s": float(dt.min()) if len(dt) else np.nan,
        "dt_median_s": float(np.median(dt)) if len(dt) else np.nan,
        "dt_max_s": float(dt.max()) if len(dt) else np.nan,
        "non_positive_dt_count": int(np.sum(dt <= 0.0)) if len(dt) else 0,
        "speed_median_mps": float(np.median(speeds)) if len(speeds) else np.nan,
        "speed_max_mps": float(np.max(speeds)) if len(speeds) else np.nan,
        "unrealistic_speed_count": unrealistic_count,
        "zero_movement_steps": zero_move_count,
        "stationary_jitter_steps": jitter_count,
    }


def align_track_to_route(
    network,
    track_df,
    candidates_df,
    route_arcs,
    trip_id,
    skip_penalty=5.0,
    unconnected_penalty=20.0,
    ambiguity_tol_m=1.0,
):
    """
    Monotonically align GPS observations to reference route arc IDs using
    dynamic programming over route sequence positions.

    Preserves unmatched and ambiguous observations explicitly without fabricating labels.
    """
    trip_id_str = str(trip_id)
    arcs_df = network["arcs"]
    from_nodes = arcs_df["from_node_id"].to_numpy(dtype="int64")
    to_nodes = arcs_df["to_node_id"].to_numpy(dtype="int64")

    # Map each arc_id to its list of positions in route_arcs.
    arc_to_route_positions = {}
    for pos, arc_id in enumerate(route_arcs):
        arc_to_route_positions.setdefault(int(arc_id), []).append(pos)

    point_ids = track_df["point_id"].to_numpy(dtype="int64")

    # Collect route candidates per observation.
    obs_route_cands = {int(pid): [] for pid in point_ids}
    if not candidates_df.empty:
        for row in candidates_df.itertuples(index=False):
            arc_id = int(row.candidate_arc_id)
            if arc_id in arc_to_route_positions:
                for pos in arc_to_route_positions[arc_id]:
                    obs_route_cands[int(row.point_id)].append((pos, arc_id, float(row.distance_m)))

    # Sort each observation's route candidates by (route_pos, distance_m).
    for pid in obs_route_cands:
        obs_route_cands[pid].sort(key=lambda item: (item[0], item[2]))

    # Track greedy nearest-distance route positions to measure route-order violations.
    greedy_positions = []
    for pid in point_ids:
        cands = obs_route_cands[int(pid)]
        if cands:
            best_greedy = min(cands, key=lambda item: (item[2], item[0]))
            greedy_positions.append((int(pid), best_greedy[0]))

    greedy_order_violations = 0
    for i in range(1, len(greedy_positions)):
        if greedy_positions[i][1] < greedy_positions[i - 1][1]:
            greedy_order_violations += 1

    # Dynamic programming over observations that have >= 1 route candidate.
    active_pids = [int(pid) for pid in point_ids if obs_route_cands[int(pid)]]
    dp_choice = {}

    if active_pids:
        dp_costs = []
        dp_back = []

        first_cands = obs_route_cands[active_pids[0]]
        dp_costs.append([c[2] for c in first_cands])
        dp_back.append([-1] * len(first_cands))

        for step in range(1, len(active_pids)):
            prev_cands = obs_route_cands[active_pids[step - 1]]
            curr_cands = obs_route_cands[active_pids[step]]
            prev_costs = dp_costs[step - 1]

            curr_costs = []
            curr_back = []
            for pos_c, arc_c, dist_c in curr_cands:
                best_cost = np.inf
                best_prev_idx = -1
                for j, (pos_p, arc_p, _) in enumerate(prev_cands):
                    if pos_c < pos_p:
                        continue  # Enforce monotonic route order.
                    step_skip = pos_c - pos_p
                    trans_cost = skip_penalty * step_skip
                    if step_skip > 0:
                        # Check directed connectivity if advancing by 1 or more route arcs.
                        connected = (to_nodes[arc_p] == from_nodes[arc_c])
                        if not connected and step_skip == 1:
                            trans_cost += unconnected_penalty
                        elif not connected and step_skip > 1:
                            # Multi-arc skip along route; penalize proportionally.
                            trans_cost += 0.5 * unconnected_penalty
                    total = prev_costs[j] + dist_c + trans_cost
                    if total < best_cost:
                        best_cost = total
                        best_prev_idx = j
                curr_costs.append(best_cost)
                curr_back.append(best_prev_idx)
            dp_costs.append(curr_costs)
            dp_back.append(curr_back)

        # Backtrack best monotonic path.
        last_costs = dp_costs[-1]
        best_last_idx = int(np.argmin(last_costs))
        if np.isfinite(last_costs[best_last_idx]):
            idx = best_last_idx
            for step in range(len(active_pids) - 1, -1, -1):
                dp_choice[active_pids[step]] = obs_route_cands[active_pids[step]][idx]
                idx = dp_back[step][idx]

    # Build per-observation alignment table with explicit unmatched & ambiguous states.
    records = []
    for i, pid in enumerate(point_ids):
        pid_int = int(pid)
        cands = obs_route_cands[pid_int]
        if not cands or pid_int not in dp_choice:
            records.append({
                "trip_id": trip_id_str,
                "point_id": pid_int,
                "alignment_status": "unmatched",
                "aligned_arc_id": pd.NA,
                "aligned_route_pos": pd.NA,
                "aligned_distance_m": np.nan,
                "route_candidate_count": len(cands),
                "monotonic_close_count": 0,
            })
            continue

        chosen_pos, chosen_arc, chosen_dist = dp_choice[pid_int]

        # Determine monotonic bounds from neighbors to check ambiguity within [prev_pos, next_pos].
        prev_bound = 0
        for back_pid in point_ids[:i][::-1]:
            if int(back_pid) in dp_choice:
                prev_bound = dp_choice[int(back_pid)][0]
                break
        next_bound = len(route_arcs) - 1
        for fwd_pid in point_ids[i + 1:]:
            if int(fwd_pid) in dp_choice:
                next_bound = dp_choice[int(fwd_pid)][0]
                break

        monotonic_cands = [
            c for c in cands if prev_bound <= c[0] <= next_bound
        ]
        # Distinct arc_ids within ambiguity_tol_m of the chosen distance.
        close_arcs = {
            c[1] for c in monotonic_cands if abs(c[2] - chosen_dist) <= ambiguity_tol_m
        }

        if len(close_arcs) > 1:
            records.append({
                "trip_id": trip_id_str,
                "point_id": pid_int,
                "alignment_status": "ambiguous",
                "aligned_arc_id": pd.NA,
                "aligned_route_pos": pd.NA,
                "aligned_distance_m": chosen_dist,
                "route_candidate_count": len(cands),
                "monotonic_close_count": len(close_arcs),
            })
        else:
            records.append({
                "trip_id": trip_id_str,
                "point_id": pid_int,
                "alignment_status": "matched",
                "aligned_arc_id": int(chosen_arc),
                "aligned_route_pos": int(chosen_pos),
                "aligned_distance_m": chosen_dist,
                "route_candidate_count": len(cands),
                "monotonic_close_count": 1,
            })

    alignment_df = pd.DataFrame(records)
    alignment_df["aligned_arc_id"] = alignment_df["aligned_arc_id"].astype("Int64")
    alignment_df["aligned_route_pos"] = alignment_df["aligned_route_pos"].astype("Int64")

    # Validate monotonic order and directed connectivity of matched observations.
    matched = alignment_df[alignment_df["alignment_status"] == "matched"]
    matched_pos = matched["aligned_route_pos"].to_numpy(dtype="int64")
    matched_arcs = matched["aligned_arc_id"].to_numpy(dtype="int64")
    aligned_order_violations = int(np.sum(np.diff(matched_pos) < 0)) if len(matched_pos) > 1 else 0

    conn_breaks = 0
    for j in range(1, len(matched_arcs)):
        a_prev = matched_arcs[j - 1]
        a_curr = matched_arcs[j]
        if a_prev != a_curr and to_nodes[a_prev] != from_nodes[a_curr]:
            conn_breaks += 1

    stats = {
        "trip_id": trip_id_str,
        "total_observations": int(len(point_ids)),
        "matched_observations": int((alignment_df["alignment_status"] == "matched").sum()),
        "ambiguous_observations": int((alignment_df["alignment_status"] == "ambiguous").sum()),
        "unmatched_observations": int((alignment_df["alignment_status"] == "unmatched").sum()),
        "greedy_route_order_violations": greedy_order_violations,
        "aligned_route_order_violations": aligned_order_violations,
        "matched_connectivity_breaks": conn_breaks,
    }
    return alignment_df, stats


def evaluate_alignment_on_trips(train_dir, trip_ids, radius_m=50, output_dir=None):
    """
    Run monotonic alignment, route connectivity validation, and speed/timestamp diagnostics
    across multiple training trips.
    """
    train_path = Path(train_dir)
    all_alignments = []
    summary_rows = []

    for tid in trip_ids:
        trip_path = train_path / tid
        network = load_road_network(trip_path / f"{tid}.nodes", trip_path / f"{tid}.arcs")
        track_df = load_track(trip_path / f"{tid}.track")
        route_arcs = load_route(trip_path / f"{tid}.route")
        cands_df = generate_candidates(network, track_df, trip_id=tid, radius_m=radius_m)

        route_val = validate_route_connectivity(network, route_arcs)
        kin_stats = analyze_track_kinematics(network, track_df)
        align_df, align_stats = align_track_to_route(
            network, track_df, cands_df, route_arcs, trip_id=tid
        )

        combined_row = {**align_stats, **route_val, **kin_stats}
        summary_rows.append(combined_row)
        all_alignments.append(align_df)

    alignments_df = pd.concat(all_alignments, ignore_index=True) if all_alignments else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        alignments_df.to_csv(out_path / "train_alignment_labels.csv", index=False)
        summary_df.to_csv(out_path / "train_alignment_summary.csv", index=False)

    return alignments_df, summary_df


if __name__ == "__main__":
    train_root = Path("Student folder/train")
    sample_trips = sorted([d.name for d in train_root.iterdir() if d.is_dir()])[:5]
    alignments, summary = evaluate_alignment_on_trips(
        train_dir=train_root,
        trip_ids=sample_trips,
        radius_m=50,
        output_dir="outputs/alignment",
    )
    print("Alignment & Kinematics Summary (5 training trips, 50m radius):")
    cols = [
        "trip_id",
        "total_observations",
        "matched_observations",
        "ambiguous_observations",
        "unmatched_observations",
        "greedy_route_order_violations",
        "aligned_route_order_violations",
        "directed_breaks",
        "dt_median_s",
        "speed_median_mps",
        "speed_max_mps",
        "unrealistic_speed_count",
        "stationary_jitter_steps",
    ]
    print(summary[cols].to_string(index=False))
