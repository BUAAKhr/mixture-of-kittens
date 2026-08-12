from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = []
    for path in args.inputs:
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())

    measurements = [
        record for record in records if record.get("kind") == "pk_hierarchical_bf16"
    ]
    stage_models = [
        record
        for record in records
        if record.get("kind") == "pk_hierarchical_bf16_stage_cost_model"
    ]
    flat_by_trial: dict[tuple[str, int, int, int, int], float] = {}
    for record in measurements:
        if record.get("mode") == "flat" and "median_ms" in record:
            key = (
                str(record["backend_label"]),
                int(record.get("trial_id", 0)),
                int(record["m"]),
                int(record["k"]),
                int(record["n"]),
            )
            if key in flat_by_trial:
                raise ValueError(f"duplicate flat record for {key}")
            flat_by_trial[key] = float(record["median_ms"])

    table = []
    for record in measurements:
        if "median_ms" not in record:
            continue
        backend = str(record["backend_label"])
        trial_id = int(record.get("trial_id", 0))
        median = float(record["median_ms"])
        flat = flat_by_trial.get(
            (
                backend,
                trial_id,
                int(record["m"]),
                int(record["k"]),
                int(record["n"]),
            )
        )
        pipeline = record.get("pipeline") or {}
        config = pipeline.get("config") or {}
        layout = pipeline.get("layout") or {}
        table.append(
            {
                "backend": backend,
                "trial_id": trial_id,
                "mode": record["mode"],
                "m": record["m"],
                "k": record["k"],
                "n": record["n"],
                "window_tiles": config.get("window_tiles"),
                "window_kib": (
                    config.get("window_tiles", 0) * layout.get("tile_bytes", 0) // 1024
                    if config and layout
                    else None
                ),
                "max_inflight": config.get("max_inflight"),
                "wire_mib_per_rank": (
                    layout.get("wire_bytes", 0) / (1024 * 1024)
                    if layout
                    else None
                ),
                "median_ms": median,
                "p95_ms": record.get("p95_ms"),
                "delta_vs_flat_percent": (
                    100.0 * (median - flat) / flat if flat else None
                ),
                "max_abs_diff": record["correctness"]["max_abs_diff"],
            }
        )
    table.sort(
        key=lambda item: (
            item["backend"],
            item["trial_id"],
            item["median_ms"],
        )
    )
    grouped: dict[tuple[object, ...], dict[int, dict[str, object]]] = defaultdict(dict)
    for row in table:
        key = (
            row["backend"],
            row["mode"],
            row["m"],
            row["k"],
            row["n"],
            row["window_tiles"],
            row["max_inflight"],
        )
        trial_id = int(row["trial_id"])
        if trial_id in grouped[key]:
            raise ValueError(
                f"duplicate configuration record for {key}, trial {trial_id}"
            )
        grouped[key][trial_id] = row
    configurations = []
    for key, rows_by_trial in grouped.items():
        trial_ids = sorted(rows_by_trial)
        rows = [rows_by_trial[trial_id] for trial_id in trial_ids]
        medians = [float(row["median_ms"]) for row in rows]
        deltas = [
            float(row["delta_vs_flat_percent"])
            for row in rows
            if row["delta_vs_flat_percent"] is not None
        ]
        configurations.append(
            {
                "backend": key[0],
                "mode": key[1],
                "m": key[2],
                "k": key[3],
                "n": key[4],
                "window_tiles": key[5],
                "max_inflight": key[6],
                "trial_ids": trial_ids,
                "fresh_process_trials": len(trial_ids),
                "complete_three_trial_result": len(trial_ids) == 3,
                "median_of_trial_medians_ms": statistics.median(medians),
                "median_delta_vs_paired_flat_percent": (
                    statistics.median(deltas) if len(deltas) == len(rows) else None
                ),
                "trial_medians_ms": medians,
            }
        )
    configurations.sort(
        key=lambda item: (
            item["backend"],
            item["m"],
            item["k"],
            item["n"],
            item["median_of_trial_medians_ms"],
        )
    )
    complete_configurations = [
        item for item in configurations if item["complete_three_trial_result"]
    ]
    summary = {
        "kind": "pk_hierarchical_bf16_summary",
        "trial_rows": table,
        "configurations": configurations,
        "best_by_backend": {
            backend: min(
                (
                    row
                    for row in complete_configurations
                    if row["backend"] == backend
                ),
                key=lambda row: row["median_of_trial_medians_ms"],
            )
            for backend in sorted(
                {row["backend"] for row in complete_configurations}
            )
        },
        "stage_models": stage_models,
    }
    payload = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
