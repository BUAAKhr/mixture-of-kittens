from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import IntEnum


MAX_COUNTER_VALUE = (1 << 64) - 1


class ProtocolError(RuntimeError):
    pass


class StaleEpochError(ProtocolError):
    pass


class DuplicateCompletionError(ProtocolError):
    pass


class OutOfOrderCompletionError(ProtocolError):
    pass


class Phase(IntEnum):
    EMPTY = 0
    LOCAL_READY = 1
    INTER_NODE_READY = 2
    BROADCAST_READY = 3
    CONSUMED = 4


@dataclass(frozen=True)
class ChunkDesc:
    epoch: int
    chunk_id: int
    offset_bytes: int
    size_bytes: int
    owner_local_rank: int

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.chunk_id < 0 or self.offset_bytes < 0:
            raise ValueError("epoch, chunk id, and offset must be non-negative")
        if self.size_bytes <= 0:
            raise ValueError("chunk size must be positive")
        if self.owner_local_rank < 0:
            raise ValueError("owner_local_rank must be non-negative")

    @property
    def end_bytes(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True)
class ChunkPlan:
    total_bytes: int
    chunk_bytes: int
    local_world_size: int

    def __post_init__(self) -> None:
        if self.total_bytes <= 0 or self.chunk_bytes <= 0:
            raise ValueError("total_bytes and chunk_bytes must be positive")
        if self.local_world_size <= 0:
            raise ValueError("local_world_size must be positive")

    @property
    def num_chunks(self) -> int:
        return (self.total_bytes + self.chunk_bytes - 1) // self.chunk_bytes

    def chunks(self, epoch: int) -> tuple[ChunkDesc, ...]:
        result = []
        for chunk_id in range(self.num_chunks):
            offset = chunk_id * self.chunk_bytes
            result.append(
                ChunkDesc(
                    epoch=epoch,
                    chunk_id=chunk_id,
                    offset_bytes=offset,
                    size_bytes=min(self.chunk_bytes, self.total_bytes - offset),
                    owner_local_rank=chunk_id % self.local_world_size,
                )
            )
        return tuple(result)


def completion_token(epoch: int, chunk_id: int, phase: Phase) -> int:
    """Pack epoch/chunk/phase into the 64-bit value used by GPU-visible slots."""

    if not 0 <= epoch < (1 << 31):
        raise ValueError("epoch must fit in 31 bits")
    if not 0 <= chunk_id < (1 << 29):
        raise ValueError("chunk_id must fit in 29 bits")
    value = (epoch << 33) | (chunk_id << 4) | int(phase)
    if value > MAX_COUNTER_VALUE:
        raise ValueError("completion token overflow")
    return value


class CompletionBoard:
    """Host reference model for exactly-once per-chunk phase publication."""

    def __init__(self, num_chunks: int, epoch: int = 0) -> None:
        if num_chunks <= 0:
            raise ValueError("num_chunks must be positive")
        self._num_chunks = num_chunks
        self._epoch = epoch
        self._phases = [Phase.EMPTY] * num_chunks
        self._condition = threading.Condition()

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset(self, epoch: int) -> None:
        with self._condition:
            if epoch <= self._epoch:
                raise StaleEpochError(
                    f"new epoch {epoch} must be greater than {self._epoch}"
                )
            self._epoch = epoch
            self._phases = [Phase.EMPTY] * self._num_chunks
            self._condition.notify_all()

    def phase(self, chunk: ChunkDesc) -> Phase:
        with self._condition:
            self._validate(chunk)
            return self._phases[chunk.chunk_id]

    def publish(self, chunk: ChunkDesc, phase: Phase) -> int:
        if phase == Phase.EMPTY:
            raise ValueError("EMPTY cannot be published")
        with self._condition:
            self._validate(chunk)
            current = self._phases[chunk.chunk_id]
            if phase == current:
                raise DuplicateCompletionError(
                    f"chunk {chunk.chunk_id} phase {phase.name} was published twice"
                )
            if int(phase) != int(current) + 1:
                raise OutOfOrderCompletionError(
                    f"chunk {chunk.chunk_id}: {current.name} -> {phase.name}"
                )
            self._phases[chunk.chunk_id] = phase
            self._condition.notify_all()
            return completion_token(chunk.epoch, chunk.chunk_id, phase)

    def wait(
        self, chunk: ChunkDesc, phase: Phase, timeout_s: float = 30.0
    ) -> Phase:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        deadline = time.monotonic() + timeout_s
        with self._condition:
            self._validate(chunk)
            while self._phases[chunk.chunk_id] < phase:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"chunk {chunk.chunk_id} did not reach {phase.name} "
                        f"in epoch {chunk.epoch}"
                    )
                self._condition.wait(remaining)
                self._validate(chunk)
            return self._phases[chunk.chunk_id]

    def snapshot(self) -> tuple[Phase, ...]:
        with self._condition:
            return tuple(self._phases)

    def _validate(self, chunk: ChunkDesc) -> None:
        if chunk.epoch != self._epoch:
            raise StaleEpochError(
                f"chunk epoch {chunk.epoch} does not match active epoch {self._epoch}"
            )
        if not 0 <= chunk.chunk_id < self._num_chunks:
            raise ProtocolError(f"chunk {chunk.chunk_id} is outside the plan")
