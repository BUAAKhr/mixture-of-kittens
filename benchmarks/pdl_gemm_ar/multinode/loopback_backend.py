from __future__ import annotations

from collections.abc import Sequence

from .protocol import ChunkDesc, CompletionBoard, Phase
from .topology import RankTopology
from .transport import HierarchicalTransport, TransportBuffers


class LoopbackTransport(HierarchicalTransport):
    """Dependency-free semantic backend used before touching NCCL or RDMA."""

    def __init__(self) -> None:
        self._topology: RankTopology | None = None
        self._buffers: TransportBuffers | None = None
        self._chunks: tuple[ChunkDesc, ...] = ()
        self._board: CompletionBoard | None = None

    def prepare(
        self,
        topology: RankTopology,
        buffers: TransportBuffers,
        chunks: Sequence[ChunkDesc],
    ) -> None:
        if not chunks:
            raise ValueError("at least one chunk is required")
        epoch = chunks[0].epoch
        if any(chunk.epoch != epoch for chunk in chunks):
            raise ValueError("all prepared chunks must share an epoch")
        self._topology = topology
        self._buffers = buffers
        self._chunks = tuple(chunks)
        self._board = CompletionBoard(len(chunks), epoch)

    def local_reduce(self, chunk: ChunkDesc) -> None:
        self._copy(chunk)
        self._require_board().publish(chunk, Phase.LOCAL_READY)

    def inter_node_exchange(self, chunk: ChunkDesc) -> None:
        board = self._require_board()
        board.wait(chunk, Phase.LOCAL_READY)
        board.publish(chunk, Phase.INTER_NODE_READY)

    def local_broadcast(self, chunk: ChunkDesc) -> None:
        board = self._require_board()
        board.wait(chunk, Phase.INTER_NODE_READY)
        board.publish(chunk, Phase.BROADCAST_READY)
        board.publish(chunk, Phase.CONSUMED)

    def wait(self, epoch: int, timeout_s: float = 30.0) -> None:
        for chunk in self._chunks:
            if chunk.epoch != epoch:
                raise ValueError("wait epoch does not match prepared chunks")
            self._require_board().wait(chunk, Phase.CONSUMED, timeout_s)

    def reset(self, epoch: int) -> None:
        board = self._require_board()
        board.reset(epoch)
        self._chunks = tuple(
            ChunkDesc(
                epoch,
                chunk.chunk_id,
                chunk.offset_bytes,
                chunk.size_bytes,
                chunk.owner_local_rank,
            )
            for chunk in self._chunks
        )

    def close(self) -> None:
        self._topology = None
        self._buffers = None
        self._chunks = ()
        self._board = None

    @property
    def board(self) -> CompletionBoard:
        return self._require_board()

    def _copy(self, chunk: ChunkDesc) -> None:
        if self._buffers is None:
            raise RuntimeError("transport was not prepared")
        source = memoryview(self._buffers.source).cast("B")
        destination = memoryview(self._buffers.destination).cast("B")
        destination[chunk.offset_bytes : chunk.end_bytes] = source[
            chunk.offset_bytes : chunk.end_bytes
        ]

    def _require_board(self) -> CompletionBoard:
        if self._board is None:
            raise RuntimeError("transport was not prepared")
        return self._board
