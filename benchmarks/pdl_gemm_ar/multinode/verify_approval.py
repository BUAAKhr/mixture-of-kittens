from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from .run_environment import communication_environment


def git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    # Preserve the leading status marker from `git submodule status`.
    return completed.stdout.rstrip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_sha256(record: dict[str, object]) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pdl_extension_candidates() -> list[Path]:
    return sorted(Path("benchmarks/pdl_gemm_ar").glob("_C*.so"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("approval", type=Path)
    parser.add_argument("--current-preflight", type=Path, required=True)
    parser.add_argument("--backend-label", required=True)
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--max-age-seconds", type=int, default=3600)
    args = parser.parse_args()

    record = json.loads(args.approval.read_text(encoding="utf-8"))
    current_preflight = json.loads(
        args.current_preflight.read_text(encoding="utf-8")
    )
    failures = []
    if record.get("kind") != "hierarchical_preflight_comparison":
        failures.append("wrong approval record kind")
    if record.get("approved_to_run") is not True:
        failures.append("preflight comparison did not approve the run")
    if record.get("backend_label") != args.backend_label:
        failures.append("backend label differs from the approved preflight")
    if socket.gethostname() not in record.get("hosts", []):
        failures.append("current host is absent from the approved preflight")
    if (record.get("host_by_node_rank") or {}).get(str(args.node_rank)) != (
        socket.gethostname()
    ):
        failures.append("current host does not match the approved node rank")
    approved_preflight_sha256 = (
        record.get("preflight_sha256_by_node_rank") or {}
    ).get(str(args.node_rank))
    if record_sha256(current_preflight) != approved_preflight_sha256:
        failures.append("current preflight snapshot differs from approval")
    if record.get("git_head") != git_output("rev-parse", "HEAD"):
        failures.append("Git HEAD differs from the approved preflight")
    if record.get("submodules") != git_output("submodule", "status"):
        failures.append("submodule state differs from the approved preflight")
    if git_output("status", "--short"):
        failures.append("worktree is not clean")
    created_at = record.get("created_at_unix")
    if (
        not isinstance(created_at, (int, float))
        or args.max_age_seconds <= 0
        or not 0 <= time.time() - created_at <= args.max_age_seconds
    ):
        failures.append("preflight approval is missing, future-dated, or expired")
    approved_environment = record.get("approved_environment")
    if approved_environment != communication_environment():
        failures.append("communication environment differs from preflight")
    extension_candidates = pdl_extension_candidates()
    if len(extension_candidates) != 1:
        failures.append("expected exactly one benchmarks/pdl_gemm_ar/_C*.so")
    elif file_sha256(extension_candidates[0]) != record.get(
        "pdl_extension_sha256"
    ):
        failures.append("PDL extension binary differs from preflight")
    if args.backend_label != "nccl-native":
        plugin_value = os.environ.get("NCCL_NET_PLUGIN")
        plugin_path = Path(plugin_value) if plugin_value else None
        if (
            plugin_path is None
            or not plugin_path.is_absolute()
            or not plugin_path.is_file()
        ):
            failures.append(
                "NCCL_NET_PLUGIN must remain an existing absolute path"
            )
        elif file_sha256(plugin_path) != record.get("plugin_sha256"):
            failures.append("NCCL_NET_PLUGIN content differs from preflight")
    broker_artifacts = [Path("/dev/shm/kittens_broker_shm")]
    broker_artifacts.extend(Path("/tmp").glob("kittens_broker.sock*"))
    stale = [str(path) for path in broker_artifacts if path.exists()]
    if stale:
        failures.append("stale KittensBroker artifacts: " + ", ".join(stale))
    if failures:
        raise SystemExit("approval rejected: " + "; ".join(failures))


if __name__ == "__main__":
    main()
