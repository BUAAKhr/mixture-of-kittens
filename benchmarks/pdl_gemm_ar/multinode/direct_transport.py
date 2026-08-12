from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class CompletionKind(str, Enum):
    LOCAL = "local"
    REMOTE = "remote"


@dataclass(frozen=True)
class RegisteredRegion:
    tensor: torch.Tensor
    registration_id: int
    local_key: int
    address: int
    length: int


@dataclass(frozen=True)
class RemoteRegion:
    address: int
    remote_key: int
    length: int


@dataclass(frozen=True)
class Transfer:
    transfer_id: int
    first_slot: int
    slot_count: int
    epoch: int
    completion: CompletionKind


class DirectOwnerTransport(ABC):
    """Interface-only boundary for the post-NCCL UCCL/RDMA experiment.

    The first implementation is intentionally two-node only.  Each owner lane
    writes its local BF16 shard into a peer receive buffer, waits for local and
    remote completion, adds the remote shard locally, then calls the existing
    PK unpack kernel.  Four or more nodes require an explicit ring/tree design
    and must not silently reuse this pairwise contract.

    No implementation is registered with the BF16 benchmark yet.  The caller
    must separately allocate/register a peer receive buffer and perform the
    BF16 local-plus-remote reduction before publishing GPU readiness.  Keeping
    those missing steps explicit prevents this interface from being mistaken
    for a runnable direct-UCCL backend.
    """

    @abstractmethod
    def register_region(self, tensor: torch.Tensor) -> RegisteredRegion:
        pass

    @abstractmethod
    def exchange_remote_region(
        self,
        local: RegisteredRegion,
        lane_group: Any,
    ) -> RemoteRegion:
        pass

    @abstractmethod
    def submit_write(
        self,
        local: RegisteredRegion,
        remote: RemoteRegion,
        first_slot: int,
        slot_count: int,
        tile_bytes: int,
        epoch: int,
    ) -> Transfer:
        pass

    @abstractmethod
    def poll_completion(self, transfer: Transfer, timeout_ms: int) -> None:
        pass

    @abstractmethod
    def publish_gpu_ready(
        self,
        ready: torch.Tensor,
        first_slot: int,
        slot_count: int,
        epoch: int,
    ) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
