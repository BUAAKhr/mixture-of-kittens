from __future__ import annotations

import argparse
import json
import subprocess


def git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--streams", type=int, default=0)
    parser.add_argument("--lane-communicators", type=int, default=0)
    parser.add_argument("--sync-states", type=int, default=0)
    parser.add_argument("--progress-threads", type=int, default=0)
    args = parser.parse_args()

    insertions = 0
    deletions = 0
    files = 0
    for line in git_output(
        ["diff", "--numstat", f"{args.base}...HEAD", "--", "benchmarks/pdl_gemm_ar"]
    ).splitlines():
        added, removed, _ = line.split("\t", 2)
        if added == "-" or removed == "-":
            continue
        insertions += int(added)
        deletions += int(removed)
        files += 1
    record = {
        "kind": "hierarchical_complexity",
        "base": args.base,
        "head": git_output(["rev-parse", "HEAD"]),
        "files_changed": files,
        "insertions": insertions,
        "deletions": deletions,
        "synchronization_states": args.sync_states,
        "cuda_streams": args.streams,
        "lane_communicators": args.lane_communicators,
        "progress_threads": args.progress_threads,
    }
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
