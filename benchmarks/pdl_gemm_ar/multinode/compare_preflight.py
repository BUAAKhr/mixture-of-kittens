from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def nested(record: dict[str, object], *keys: str) -> object:
    value: object = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def record_sha256(record: dict[str, object]) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("node0", type=Path)
    parser.add_argument("node1", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = [
        json.loads(args.node0.read_text(encoding="utf-8")),
        json.loads(args.node1.read_text(encoding="utf-8")),
    ]
    checks: dict[str, bool] = {}
    checks["node_rank_order"] = [
        record.get("node_rank") for record in records
    ] == [0, 1]
    checks["different_hosts"] = records[0].get("hostname") != records[1].get("hostname")
    checks["matching_backend"] = (
        records[0].get("backend_label") == records[1].get("backend_label")
    )
    checks["matching_git_head"] = (
        nested(records[0], "git_head", "output")
        == nested(records[1], "git_head", "output")
        and bool(nested(records[0], "git_head", "output"))
    )
    checks["matching_submodules"] = (
        nested(records[0], "submodules", "output")
        == nested(records[1], "submodules", "output")
        and bool(nested(records[0], "submodules", "output"))
    )
    checks["submodules_at_index"] = all(
        bool(output := nested(record, "submodules", "output"))
        and all(line.startswith(" ") for line in str(output).splitlines())
        for record in records
    )
    checks["clean_worktrees"] = all(
        nested(record, "git_status", "output") == "" for record in records
    )
    checks["clean_kittens_broker"] = all(
        record.get("kittens_broker_clean") is True for record in records
    )
    checks["eight_gpus_per_node"] = all(
        nested(record, "torch", "device_count") == 8 for record in records
    )
    checks["all_h100"] = all(
        all("H100" in name for name in nested(record, "torch", "devices") or [])
        for record in records
    )
    checks["full_nvlink_mesh"] = all(
        record.get("full_nvlink_mesh") is True for record in records
    )
    checks["torch_cuda_available"] = all(
        nested(record, "torch", "available") is True
        and nested(record, "torch", "cuda_available") is True
        for record in records
    )
    checks["matching_torch_version"] = (
        nested(records[0], "torch", "version")
        == nested(records[1], "torch", "version")
        and bool(nested(records[0], "torch", "version"))
    )
    checks["matching_torch_cuda_version"] = (
        nested(records[0], "torch", "cuda_version")
        == nested(records[1], "torch", "cuda_version")
        and bool(nested(records[0], "torch", "cuda_version"))
    )
    checks["matching_python_version"] = (
        nested(records[0], "python", "version")
        == nested(records[1], "python", "version")
        and bool(nested(records[0], "python", "version"))
    )
    checks["matching_python_executable"] = (
        nested(records[0], "python", "executable")
        == nested(records[1], "python", "executable")
        and bool(nested(records[0], "python", "executable"))
    )
    checks["matching_nvidia_driver"] = (
        nested(records[0], "nvidia_driver", "output")
        == nested(records[1], "nvidia_driver", "output")
        and bool(nested(records[0], "nvidia_driver", "output"))
    )
    checks["matching_gpu_limits"] = (
        nested(records[0], "gpu_limits", "output")
        == nested(records[1], "gpu_limits", "output")
        and bool(nested(records[0], "gpu_limits", "output"))
    )
    checks["matching_pdl_extension"] = (
        all(
            nested(record, "pdl_extension", "unique") is True
            and bool(nested(record, "pdl_extension", "sha256"))
            for record in records
        )
        and nested(records[0], "pdl_extension", "sha256")
        == nested(records[1], "pdl_extension", "sha256")
    )
    checks["ib_devices_available"] = all(
        nested(record, "ib_devices", "available") is True
        and nested(record, "ib_devices", "returncode") == 0
        and bool(nested(record, "ib_devices", "output"))
        for record in records
    )
    checks["ib_net_mapping_available"] = all(
        nested(record, "ibdev2netdev", "available") is True
        and nested(record, "ibdev2netdev", "returncode") == 0
        and bool(nested(record, "ibdev2netdev", "output"))
        for record in records
    )
    checks["matching_communication_environment"] = (
        records[0].get("communication_environment")
        == records[1].get("communication_environment")
        and bool(records[0].get("communication_environment"))
    )
    backend_label = str(records[0].get("backend_label"))
    checks["explicit_socket_interface"] = bool(
        nested(records[0], "environment", "NCCL_SOCKET_IFNAME")
    )
    checks["explicit_gid_index"] = bool(
        nested(records[0], "environment", "NCCL_IB_GID_INDEX")
    )
    checks["explicit_ib_hca"] = bool(
        nested(records[0], "environment", "NCCL_IB_HCA")
    )
    for name in (
        "MASTER_ADDR",
        "MASTER_PORT",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "TORCH_NCCL_BLOCKING_WAIT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "OMP_NUM_THREADS",
    ):
        checks[f"explicit_{name.lower()}"] = bool(
            nested(records[0], "communication_environment", name)
        )
    if backend_label == "nccl-native":
        checks["backend_plugin_selection"] = all(
            nested(record, "environment", "NCCL_NET_PLUGIN") is None
            and nested(record, "environment", "UCCL_PLUGIN_PATH") is None
            for record in records
        )
    else:
        checks["backend_plugin_selection"] = all(
            nested(record, "selected_nccl_net_plugin", "is_absolute") is True
            and nested(record, "selected_nccl_net_plugin", "exists") is True
            and bool(nested(record, "selected_nccl_net_plugin", "sha256"))
            and nested(record, "environment", "UCCL_PLUGIN_PATH")
            == nested(record, "environment", "NCCL_NET_PLUGIN")
            for record in records
        ) and (
            nested(records[0], "selected_nccl_net_plugin", "sha256")
            == nested(records[1], "selected_nccl_net_plugin", "sha256")
        )

    result = {
        "kind": "hierarchical_preflight_comparison",
        "checks": checks,
        "approved_to_run": all(checks.values()),
        "hosts": [record.get("hostname") for record in records],
        "host_by_node_rank": {
            str(record.get("node_rank")): record.get("hostname")
            for record in records
        },
        "preflight_sha256_by_node_rank": {
            str(record.get("node_rank")): record_sha256(record)
            for record in records
        },
        "backend_label": backend_label,
        "git_head": nested(records[0], "git_head", "output"),
        "submodules": nested(records[0], "submodules", "output"),
        "approved_environment": records[0].get("communication_environment"),
        "plugin_sha256": nested(
            records[0], "selected_nccl_net_plugin", "sha256"
        ),
        "pdl_extension_sha256": nested(
            records[0], "pdl_extension", "sha256"
        ),
        "created_at_unix": time.time(),
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["approved_to_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
