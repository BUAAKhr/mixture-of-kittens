from __future__ import annotations

import os
from collections.abc import Mapping


EXACT_ENVIRONMENT_NAMES = {
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "MASTER_ADDR",
    "MASTER_PORT",
    "OMP_NUM_THREADS",
}
ENVIRONMENT_PREFIXES = ("NCCL_", "TORCH_NCCL_", "UCCL_")


def communication_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environment is None else environment
    return {
        name: value
        for name, value in sorted(source.items())
        if name in EXACT_ENVIRONMENT_NAMES
        or name.startswith(ENVIRONMENT_PREFIXES)
    }
