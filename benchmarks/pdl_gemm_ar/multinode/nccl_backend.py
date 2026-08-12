from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from .protocol import ChunkDesc
from .topology import RankTopology


@dataclass
class NcclGroups:
    local_reduce: Any
    local_broadcast: Any
    lane_channels: tuple[tuple[Any, ...], ...]

    def lane(self, local_rank: int, channel: int = 0) -> Any:
        return self.lane_channels[local_rank][channel]

    @property
    def num_lane_channels(self) -> int:
        return len(self.lane_channels[0])


def create_nccl_groups(
    topology: RankTopology,
    lane_channels: int = 1,
) -> NcclGroups:
    """Create every group in global order, returning the groups for this rank."""

    if lane_channels <= 0:
        raise ValueError("lane_channels must be positive")

    local_reduce = None
    local_broadcast = None
    for node_rank in range(topology.num_nodes):
        ranks = list(topology.ranks_on_node(node_rank))
        reduce_group = dist.new_group(ranks, backend="nccl")
        broadcast_group = dist.new_group(ranks, backend="nccl")
        if node_rank == topology.node_rank:
            local_reduce = reduce_group
            local_broadcast = broadcast_group

    lanes = []
    for local_rank in range(topology.local_world_size):
        lanes.append(
            tuple(
                dist.new_group(
                    list(topology.ranks_in_lane(local_rank)), backend="nccl"
                )
                for _ in range(lane_channels)
            )
        )

    if local_reduce is None or local_broadcast is None:
        raise RuntimeError("failed to create local NCCL groups")
    return NcclGroups(local_reduce, local_broadcast, tuple(lanes))


class NcclHierarchicalCollective:
    """BF16 reference implementation of the three-stage hierarchy."""

    def __init__(self, topology: RankTopology, groups: NcclGroups) -> None:
        self.topology = topology
        self.groups = groups

    def flat_all_reduce(self, tensor: torch.Tensor) -> None:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def sequential(self, tensor: torch.Tensor, chunks: tuple[ChunkDesc, ...]) -> None:
        for chunk in chunks:
            view = self._view(tensor, chunk)
            owner = self.topology.owner_rank(chunk.chunk_id)
            dist.reduce(view, dst=owner, op=dist.ReduceOp.SUM, group=self.groups.local_reduce)
            if self.topology.local_rank == chunk.owner_local_rank:
                dist.all_reduce(
                    view,
                    op=dist.ReduceOp.SUM,
                    group=self.groups.lane(chunk.owner_local_rank),
                )
            dist.broadcast(view, src=owner, group=self.groups.local_broadcast)

    def windowed(
        self,
        tensor: torch.Tensor,
        chunks: tuple[ChunkDesc, ...],
        depth: int,
    ) -> None:
        if depth <= 0:
            raise ValueError("window depth must be positive")
        for first in range(0, len(chunks), depth):
            window = chunks[first : first + depth]
            local_works = []
            for chunk in window:
                local_works.append(
                    dist.reduce(
                        self._view(tensor, chunk),
                        dst=self.topology.owner_rank(chunk.chunk_id),
                        op=dist.ReduceOp.SUM,
                        group=self.groups.local_reduce,
                        async_op=True,
                    )
                )

            inter_works: list[Any | None] = []
            for chunk, work in zip(window, local_works, strict=True):
                work.wait()
                if self.topology.local_rank == chunk.owner_local_rank:
                    inter_works.append(
                        dist.all_reduce(
                            self._view(tensor, chunk),
                            op=dist.ReduceOp.SUM,
                            group=self.groups.lane(chunk.owner_local_rank),
                            async_op=True,
                        )
                    )
                else:
                    inter_works.append(None)

            broadcast_works = []
            for chunk, work in zip(window, inter_works, strict=True):
                if work is not None:
                    work.wait()
                broadcast_works.append(
                    dist.broadcast(
                        self._view(tensor, chunk),
                        src=self.topology.owner_rank(chunk.chunk_id),
                        group=self.groups.local_broadcast,
                        async_op=True,
                    )
                )
            for work in broadcast_works:
                work.wait()

    def local_reduce_only(
        self, tensor: torch.Tensor, chunks: tuple[ChunkDesc, ...]
    ) -> None:
        for chunk in chunks:
            dist.reduce(
                self._view(tensor, chunk),
                dst=self.topology.owner_rank(chunk.chunk_id),
                op=dist.ReduceOp.SUM,
                group=self.groups.local_reduce,
            )

    def inter_node_only(
        self, tensor: torch.Tensor, chunks: tuple[ChunkDesc, ...]
    ) -> None:
        for chunk in chunks:
            if self.topology.local_rank == chunk.owner_local_rank:
                dist.all_reduce(
                    self._view(tensor, chunk),
                    op=dist.ReduceOp.SUM,
                    group=self.groups.lane(chunk.owner_local_rank),
                )

    def local_broadcast_only(
        self, tensor: torch.Tensor, chunks: tuple[ChunkDesc, ...]
    ) -> None:
        for chunk in chunks:
            dist.broadcast(
                self._view(tensor, chunk),
                src=self.topology.owner_rank(chunk.chunk_id),
                group=self.groups.local_broadcast,
            )

    @staticmethod
    def _view(tensor: torch.Tensor, chunk: ChunkDesc) -> torch.Tensor:
        if not tensor.is_contiguous():
            raise ValueError("hierarchical NCCL input must be contiguous")
        item_size = tensor.element_size()
        if chunk.offset_bytes % item_size or chunk.size_bytes % item_size:
            raise ValueError("chunk boundaries must align to the tensor dtype")
        flat = tensor.view(-1)
        start = chunk.offset_bytes // item_size
        end = chunk.end_bytes // item_size
        return flat[start:end]
