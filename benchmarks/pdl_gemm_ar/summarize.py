from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_records(paths: list[Path]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("kind") == "gemm_all_reduce":
                    record["source"] = path.name
                    records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in load_records(args.paths):
        grouped[str(record["mode"])].append(record)

    print(
        "mode\truns\tmedian_of_run_medians_ms\tdelta_vs_fused_pct\t"
        "range_ms\tmedian_p95_ms"
    )
    fused_records = grouped.get("fused", [])
    fused_median = None
    if fused_records:
        fused_median = statistics.median(
            float(record["max_rank_median_ms"])
            for record in fused_records
        )
    for mode, records in sorted(grouped.items()):
        medians = [float(record["max_rank_median_ms"]) for record in records]
        p95s = [float(record["max_rank_p95_ms"]) for record in records]
        median = statistics.median(medians)
        delta = (
            100.0 * (median / fused_median - 1.0)
            if fused_median is not None
            else float("nan")
        )
        print(
            f"{mode}\t{len(records)}\t{median:.6f}\t{delta:+.2f}\t"
            f"{min(medians):.6f}-{max(medians):.6f}\t"
            f"{statistics.median(p95s):.6f}"
        )


if __name__ == "__main__":
    main()
