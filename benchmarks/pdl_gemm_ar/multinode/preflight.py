from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from .run_environment import communication_environment


def has_full_nvlink_mesh(topology_output: object, gpu_count: int) -> bool:
    if not isinstance(topology_output, str) or gpu_count <= 1:
        return False
    rows = [
        line.split()
        for line in topology_output.splitlines()
        if line.lstrip().startswith("GPU")
    ]
    gpu_rows = [row for row in rows if row and row[0][3:].isdigit()]
    if len(gpu_rows) != gpu_count:
        return False
    for row_index, row in enumerate(gpu_rows):
        if len(row) < gpu_count + 1:
            return False
        links = row[1 : gpu_count + 1]
        for column_index, link in enumerate(links):
            if column_index == row_index:
                if link != "X":
                    return False
            elif not link.startswith("NV"):
                return False
    return True


def file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, object]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"command": command, "available": False, "output": None}
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "command": command,
        "available": True,
        "returncode": completed.returncode,
        # Preserve the leading status marker from `git submodule status`.
        "output": completed.stdout.rstrip(),
        "stderr": completed.stderr.rstrip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-label", default="nccl-native")
    parser.add_argument("--node-rank", type=int, choices=(0, 1), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        import torch

        torch_record: dict[str, object] = {
            "available": True,
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # preflight must report, not hide, import failures
        torch_record = {"available": False, "error": repr(exc)}

    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "MASTER_ADDR",
        "MASTER_PORT",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
        "NCCL_DEBUG",
        "NCCL_NET",
        "NCCL_NET_PLUGIN",
        "NCCL_SOCKET_IFNAME",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_CROSS_NIC",
        "TORCH_NCCL_BLOCKING_WAIT",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "LD_LIBRARY_PATH",
        "OMP_NUM_THREADS",
        "UCCL_SOCKET_IFNAME",
        "UCCL_IB_GID_INDEX",
        "UCCL_IB_HCA",
        "UCCL_CHUNK_SIZE_KB",
        "UCCL_NUM_ENGINES",
        "UCCL_PORT_ENTROPY",
        "UCCL_PLUGIN_PATH",
    )
    broker_artifacts = [
        str(path)
        for path in (
            [Path("/dev/shm/kittens_broker_shm")]
            + sorted(Path("/tmp").glob("kittens_broker.sock*"))
        )
        if path.exists()
    ]
    selected_plugin = os.environ.get("NCCL_NET_PLUGIN")
    selected_plugin_path = Path(selected_plugin) if selected_plugin else None
    extension_candidates = sorted(
        Path("benchmarks/pdl_gemm_ar").glob("_C*.so")
    )
    nvidia_topology = command_output(["nvidia-smi", "topo", "-m"])
    device_count = (
        int(torch_record.get("device_count", 0))
        if torch_record.get("available") is True
        else 0
    )
    record = {
        "kind": "hierarchical_preflight",
        "backend_label": args.backend_label,
        "node_rank": args.node_rank,
        "hostname": socket.gethostname(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "environment": {name: os.environ.get(name) for name in environment_names},
        "communication_environment": communication_environment(),
        "torch": torch_record,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_topology": nvidia_topology,
        "full_nvlink_mesh": has_full_nvlink_mesh(
            nvidia_topology.get("output"), device_count
        ),
        "nvidia_driver": command_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ]
        ),
        "gpu_limits": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,power.limit,clocks.max.sm,clocks.max.memory,persistence_mode",
                "--format=csv,noheader",
            ]
        ),
        "ib_devices": command_output(["ibv_devices"]),
        "ibdev2netdev": command_output(["ibdev2netdev"]),
        "git_head": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "submodules": command_output(["git", "submodule", "status"]),
        "kittens_broker_artifacts": broker_artifacts,
        "kittens_broker_clean": not broker_artifacts,
        "selected_nccl_net_plugin": {
            "value": selected_plugin,
            "is_absolute": (
                selected_plugin_path.is_absolute()
                if selected_plugin_path is not None
                else False
            ),
            "exists": (
                selected_plugin_path.is_file()
                if selected_plugin_path is not None
                else False
            ),
            "sha256": file_sha256(selected_plugin_path),
        },
        "pdl_extension": {
            "candidates": [str(path) for path in extension_candidates],
            "unique": len(extension_candidates) == 1,
            "sha256": (
                file_sha256(extension_candidates[0])
                if len(extension_candidates) == 1
                else None
            ),
        },
        "nccl_plugin_candidates": [
            str(path)
            for root in os.environ.get("LD_LIBRARY_PATH", "").split(":")
            if root
            for pattern in ("libnccl-net*.so*", "libnccl-net-uccl.so*")
            for path in Path(root).glob(pattern)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
