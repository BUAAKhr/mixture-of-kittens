from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

from .protocol import ChunkDesc
from .topology import RankTopology


@dataclass
class TransportBuffers:
    source: Any
    destination: Any


class HierarchicalTransport(ABC):
    """Transport-neutral contract shared by loopback, NCCL, and UCCL."""

    @abstractmethod
    def prepare(
        self,
        topology: RankTopology,
        buffers: TransportBuffers,
        chunks: Sequence[ChunkDesc],
    ) -> None:
        pass

    @abstractmethod
    def local_reduce(self, chunk: ChunkDesc) -> None:
        pass

    @abstractmethod
    def inter_node_exchange(self, chunk: ChunkDesc) -> None:
        pass

    @abstractmethod
    def local_broadcast(self, chunk: ChunkDesc) -> None:
        pass

    @abstractmethod
    def wait(self, epoch: int, timeout_s: float = 30.0) -> None:
        pass

    @abstractmethod
    def reset(self, epoch: int) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
