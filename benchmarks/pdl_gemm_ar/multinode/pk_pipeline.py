from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.distributed as dist

from .extension import load_pdl_extension
from .nccl_backend import NcclGroups
from .topology import RankTopology


TILE_M = 128
TILE_N = 256
TILE_ELEMENTS = TILE_M * TILE_N
TILE_BYTES_BF16 = TILE_ELEMENTS * 2
LOCAL_WORLD_SIZE = 8
MAX_EPOCH = (1 << 31) - 1


@dataclass(frozen=True)
class OwnerWireLayout:
    m: int
    n: int
    local_rank: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.n <= 0:
            raise ValueError("M and N must be positive")
        if self.m % TILE_M or self.n % TILE_N:
            raise ValueError("M/N must align to the ParallelKittens output tile")
        if not 0 <= self.local_rank < LOCAL_WORLD_SIZE:
            raise ValueError("local_rank must be in the eight-GPU NVLS group")

    @property
    def row_blocks(self) -> int:
        return self.m // TILE_M

    @property
    def col_blocks(self) -> int:
        return self.n // TILE_N

    @property
    def num_tiles(self) -> int:
        return self.row_blocks * self.col_blocks

    @property
    def num_owner_slots(self) -> int:
        remaining = self.num_tiles - self.local_rank
        return 0 if remaining <= 0 else (remaining + LOCAL_WORLD_SIZE - 1) // LOCAL_WORLD_SIZE

    @property
    def wire_elements(self) -> int:
        return self.num_owner_slots * TILE_ELEMENTS

    @property
    def wire_bytes(self) -> int:
        return self.num_owner_slots * TILE_BYTES_BF16

    def windows(self, window_tiles: int) -> tuple[tuple[int, int], ...]:
        if window_tiles <= 0:
            raise ValueError("window_tiles must be positive")
        return tuple(
            (first, min(window_tiles, self.num_owner_slots - first))
            for first in range(0, self.num_owner_slots, window_tiles)
        )

    def as_dict(self) -> dict[str, int]:
        return asdict(self) | {
            "row_blocks": self.row_blocks,
            "col_blocks": self.col_blocks,
            "num_tiles": self.num_tiles,
            "num_owner_slots": self.num_owner_slots,
            "wire_elements": self.wire_elements,
            "wire_bytes": self.wire_bytes,
            "tile_bytes": TILE_BYTES_BF16,
        }


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = "pdl_tile"
    num_comp_ctas: int = 132
    num_pack_ctas: int = 20
    pack_threads: int = 1024
    unpack_threads: int = 1024
    window_tiles: int = 4
    max_inflight: int = 1
    ready_timeout_ms: int = 30_000
    instrument: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"default", "pdl_grid", "pdl_tile"}:
            raise ValueError("unsupported producer mode")
        if not 1 <= self.num_comp_ctas <= 132:
            raise ValueError("num_comp_ctas must be in 1..132")
        if not 1 <= self.num_pack_ctas <= 132:
            raise ValueError("num_pack_ctas must be in 1..132")
        if self.pack_threads not in {256, 512, 1024}:
            raise ValueError("pack_threads must be 256, 512, or 1024")
        if self.unpack_threads not in {256, 512, 1024}:
            raise ValueError("unpack_threads must be 256, 512, or 1024")
        if not 1 <= self.window_tiles <= 1024:
            raise ValueError(
                "window_tiles must be in 1..1024 because the ready wait uses "
                "one CUDA thread per owner slot"
            )
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if self.ready_timeout_ms <= 0:
            raise ValueError("ready_timeout_ms must be positive")


class PkNcclHierarchicalPipeline:
    """PK/NVLS owner-wire producer plus an inter-node NCCL lane exchange."""

    def __init__(
        self,
        topology: RankTopology,
        groups: NcclGroups,
        a: torch.Tensor,
        b: torch.Tensor,
        output: Any,
        barrier: Any,
        config: PipelineConfig,
    ) -> None:
        if topology.local_world_size != LOCAL_WORLD_SIZE:
            raise ValueError("the current PK/NVLS kernels require eight GPUs per node")
        if a.dim() != 2 or b.dim() != 2 or a.size(1) != b.size(0):
            raise ValueError("invalid GEMM tensors")
        if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
            raise ValueError("the first hierarchical implementation is BF16-only")
        self.topology = topology
        self.groups = groups
        self.a = a
        self.b = b
        self.output = output
        self.barrier = barrier
        self.config = config
        self.layout = OwnerWireLayout(a.size(0), b.size(1), topology.local_rank)
        if self.layout.num_owner_slots == 0:
            raise ValueError(
                "every local rank must own at least one tile in the GPU data path"
            )
        self.extension = load_pdl_extension()
        self.wire = torch.empty(
            self.layout.wire_elements,
            dtype=torch.bfloat16,
            device=a.device,
        )
        self.ready = torch.zeros(
            self.layout.num_owner_slots,
            dtype=torch.int32,
            device=a.device,
        )
        self.error = torch.zeros(1, dtype=torch.int32, device=a.device)
        self._epoch = 0
        windows = self.layout.windows(config.window_tiles)
        stream_count = min(config.max_inflight, max(1, len(windows)))
        if groups.num_lane_channels < stream_count:
            raise ValueError(
                "not enough NCCL lane channels for the requested in-flight depth"
            )
        self._network_streams = [
            torch.cuda.Stream(device=a.device) for _ in range(stream_count)
        ]
        self._start_event = torch.cuda.Event(enable_timing=False)
        self._stream_done = [
            torch.cuda.Event(enable_timing=False) for _ in range(stream_count)
        ]
        self._pipeline_done = torch.cuda.Event(enable_timing=False)
        self._trace_start: torch.cuda.Event | None = None
        self._producer_done: torch.cuda.Event | None = None
        self._join_done: torch.cuda.Event | None = None
        self._remote_ready: list[torch.cuda.Event] = []
        self._unpack_done: list[torch.cuda.Event] = []
        if config.instrument:
            self._trace_start = torch.cuda.Event(enable_timing=True)
            self._producer_done = torch.cuda.Event(enable_timing=True)
            self._join_done = torch.cuda.Event(enable_timing=True)

    @property
    def epoch(self) -> int:
        return self._epoch

    def reset_errors(self) -> None:
        self.error.zero_()

    def reserve_epoch(self) -> int:
        self._epoch += 1
        if self._epoch > MAX_EPOCH:
            raise OverflowError("hierarchical ready epoch exhausted")
        return self._epoch

    def enqueue_compute_only(self) -> None:
        self.extension.matmul_local_bf16(
            self.a,
            self.b,
            self.output,
            self.barrier,
            self.config.num_comp_ctas,
        )

    def enqueue_pack_only(self) -> int:
        epoch = self.reserve_epoch()
        self.extension.hierarchical_bf16_pack_only(
            self.a,
            self.b,
            self.output,
            self.barrier,
            self.wire,
            self.ready,
            self.error,
            self.config.num_pack_ctas,
            self.config.pack_threads,
            epoch,
        )
        return epoch

    def enqueue_ready_control_only(self, epoch: int) -> None:
        self._enqueue_window_stages(
            epoch=epoch,
            wait_ready=True,
            exchange=False,
            unpack=False,
            local_join=False,
            record_trace=False,
        )

    def enqueue_window_orchestration_only(self) -> None:
        self._enqueue_window_stages(
            epoch=None,
            wait_ready=False,
            exchange=False,
            unpack=False,
            local_join=False,
            record_trace=False,
        )

    def enqueue_network_only(self) -> None:
        self._enqueue_window_stages(
            epoch=None,
            wait_ready=False,
            exchange=True,
            unpack=False,
            local_join=False,
            record_trace=False,
        )

    def enqueue_unpack_only(self) -> None:
        self._enqueue_window_stages(
            epoch=None,
            wait_ready=False,
            exchange=False,
            unpack=True,
            local_join=False,
            record_trace=False,
        )

    def enqueue_local_join_only(self) -> None:
        self._enqueue_local_join()

    def enqueue(self) -> None:
        epoch = self.reserve_epoch()

        current = torch.cuda.current_stream(self.a.device)
        # Network streams must be admitted before the producer is submitted;
        # otherwise this event would serialize them behind compute+pack+reset.
        if self._trace_start is not None:
            # Record the timing origin before releasing the network streams so
            # a fast ready window cannot timestamp itself before the origin.
            self._trace_start.record(current)
        self._start_event.record(current)
        self.extension.hierarchical_bf16_producer(
            self.a,
            self.b,
            self.output,
            self.barrier,
            self.wire,
            self.ready,
            self.error,
            self.config.mode,
            self.config.num_comp_ctas,
            self.config.num_pack_ctas,
            self.config.pack_threads,
            epoch,
        )
        if self._producer_done is not None:
            self._producer_done.record(current)

        self._enqueue_window_stages(
            epoch=epoch,
            wait_ready=True,
            exchange=True,
            unpack=True,
            local_join=True,
            record_trace=self.config.instrument,
            anchor_current=False,
        )

    def _enqueue_window_stages(
        self,
        *,
        epoch: int | None,
        wait_ready: bool,
        exchange: bool,
        unpack: bool,
        local_join: bool,
        record_trace: bool,
        anchor_current: bool = True,
    ) -> None:
        if wait_ready and epoch is None:
            raise ValueError("ready waits require an epoch")
        current = torch.cuda.current_stream(self.a.device)
        if anchor_current:
            self._start_event.record(current)
        windows = self.layout.windows(self.config.window_tiles)
        if record_trace:
            self._remote_ready = [
                torch.cuda.Event(enable_timing=True) for _ in windows
            ]
            self._unpack_done = [
                torch.cuda.Event(enable_timing=True) for _ in windows
            ]
        used_streams: set[int] = set()
        for window_index, (first_slot, slot_count) in enumerate(windows):
            stream_index = window_index % len(self._network_streams)
            used_streams.add(stream_index)
            stream = self._network_streams[stream_index]
            with torch.cuda.stream(stream):
                stream.wait_event(self._start_event)
                if wait_ready:
                    self.extension.hierarchical_wait_ready(
                        self.ready,
                        self.error,
                        first_slot,
                        slot_count,
                        epoch,
                        self.config.ready_timeout_ms * 1_000_000,
                    )
                first = first_slot * TILE_ELEMENTS
                count = slot_count * TILE_ELEMENTS
                if exchange:
                    work = dist.all_reduce(
                        self.wire.narrow(0, first, count),
                        op=dist.ReduceOp.SUM,
                        group=self.groups.lane(
                            self.topology.local_rank,
                            stream_index,
                        ),
                        async_op=True,
                    )
                    # The design relies on ProcessGroupNCCL Work.wait() adding
                    # a completion dependency to the current CUDA stream.  The
                    # first approved correctness/timeline run must validate
                    # this contract for the pinned PyTorch/NCCL versions.
                    work.wait()
                    if record_trace:
                        self._remote_ready[window_index].record(stream)
                if unpack:
                    self.extension.hierarchical_bf16_unpack(
                        self.output,
                        self.wire,
                        first_slot,
                        slot_count,
                        self.config.unpack_threads,
                    )
                    if record_trace:
                        self._unpack_done[window_index].record(stream)

        for stream_index, stream in enumerate(self._network_streams):
            if stream_index in used_streams:
                self._stream_done[stream_index].record(stream)
            else:
                self._stream_done[stream_index].record(current)

        join_stream = self._network_streams[0]
        with torch.cuda.stream(join_stream):
            for event in self._stream_done:
                join_stream.wait_event(event)
            if local_join:
                self._enqueue_local_join()
                if record_trace and self._join_done is not None:
                    self._join_done.record(join_stream)
            self._pipeline_done.record(join_stream)
        current.wait_event(self._pipeline_done)

    def _enqueue_local_join(self) -> None:
        # Every local rank multicasts a disjoint owner subset.  The local join
        # uses the same system-scope GPU counter protocol as the PK kernels,
        # so completion covers all eight multicast writes rather than only an
        # unrelated NCCL control collective.
        self.extension.hierarchical_bf16_local_join(
            self.barrier,
            self.error,
            self.config.ready_timeout_ms * 1_000_000,
        )

    def check_error(self) -> None:
        global_error = self.error.clone()
        dist.all_reduce(global_error, op=dist.ReduceOp.MAX)
        code = int(global_error.item()) & 0xFFFFFFFF
        if code == 0:
            return
        slot = code & 0x0FFFFFFF
        if code & 0x40000000:
            reason = "duplicate or stale owner-slot publication"
        elif code & 0x20000000:
            reason = "GPU ready wait timed out"
        elif code & 0x10000000:
            reason = "node-local NVLS completion join timed out"
        else:
            reason = "unknown hierarchical protocol error"
        raise RuntimeError(f"{reason}: code=0x{code:08x}, slot={slot}")

    def metadata(self) -> dict[str, object]:
        return {
            "layout": self.layout.as_dict(),
            "config": asdict(self.config),
            "num_network_streams": len(self._network_streams),
            "nccl_work_wait_stream_contract": "requires_runtime_validation",
            "nvls_multicast_join": "eight_system_scope_source_slots",
        }

    def trace(self) -> dict[str, object] | None:
        if (
            self._trace_start is None
            or self._producer_done is None
            or self._join_done is None
        ):
            return None
        torch.cuda.synchronize(self.a.device)
        remote_ms = [
            float(self._trace_start.elapsed_time(event))
            for event in self._remote_ready
        ]
        unpack_ms = [
            float(self._trace_start.elapsed_time(event))
            for event in self._unpack_done
        ]
        producer_path_done_ms = float(
            self._trace_start.elapsed_time(self._producer_done)
        )
        first_remote_ms = min(remote_ms)
        last_remote_ms = max(remote_ms)
        join_done_ms = float(self._trace_start.elapsed_time(self._join_done))
        return {
            "producer_path_done_ms": producer_path_done_ms,
            "remote_ready_ms": remote_ms,
            "unpack_done_ms": unpack_ms,
            "first_remote_ready_ms": first_remote_ms,
            "last_remote_ready_ms": last_remote_ms,
            "join_done_ms": join_done_ms,
            "unpack_after_remote_by_window": [
                unpack >= remote
                for remote, unpack in zip(remote_ms, unpack_ms, strict=True)
            ],
            "join_after_all_unpacks": join_done_ms >= max(unpack_ms),
            "remote_before_producer_path_done": (
                first_remote_ms < producer_path_done_ms
            ),
            "overlap_window_ms": max(
                0.0, producer_path_done_ms - first_remote_ms
            ),
        }
