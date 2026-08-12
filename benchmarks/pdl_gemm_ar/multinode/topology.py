from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RankTopology:
    """Contiguous rank topology used by torchrun and the hierarchy benchmark."""

    rank: int
    world_size: int
    local_world_size: int

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.local_world_size <= 0:
            raise ValueError("world sizes must be positive")
        if self.world_size % self.local_world_size:
            raise ValueError("world_size must be divisible by local_world_size")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside the world")

    @classmethod
    def from_env(cls, logical_local_world_size: int | None = None) -> "RankTopology":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_world_size = logical_local_world_size
        if local_world_size is None:
            local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        return cls(rank, world_size, local_world_size)

    @property
    def num_nodes(self) -> int:
        return self.world_size // self.local_world_size

    @property
    def node_rank(self) -> int:
        return self.rank // self.local_world_size

    @property
    def local_rank(self) -> int:
        return self.rank % self.local_world_size

    @property
    def local_ranks(self) -> tuple[int, ...]:
        first = self.node_rank * self.local_world_size
        return tuple(range(first, first + self.local_world_size))

    def ranks_on_node(self, node_rank: int) -> tuple[int, ...]:
        if not 0 <= node_rank < self.num_nodes:
            raise ValueError("node_rank is outside the topology")
        first = node_rank * self.local_world_size
        return tuple(range(first, first + self.local_world_size))

    def ranks_in_lane(self, local_rank: int) -> tuple[int, ...]:
        if not 0 <= local_rank < self.local_world_size:
            raise ValueError("local_rank is outside the node")
        return tuple(
            node * self.local_world_size + local_rank
            for node in range(self.num_nodes)
        )

    def owner_local_rank(self, chunk_id: int) -> int:
        if chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        return chunk_id % self.local_world_size

    def owner_rank(self, chunk_id: int, node_rank: int | None = None) -> int:
        node = self.node_rank if node_rank is None else node_rank
        return node * self.local_world_size + self.owner_local_rank(chunk_id)

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "num_nodes": self.num_nodes,
            "node_rank": self.node_rank,
            "local_rank": self.local_rank,
            "local_ranks": self.local_ranks,
            "lane_ranks": self.ranks_in_lane(self.local_rank),
        }
