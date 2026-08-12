from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from _C import (  # type: ignore
    TKParallelTensor,
    all_reduce_split,
    fp4_pre_communication_only,
    fp4_pre_producer_only,
    matmul_local_bf16,
    matmul_all_reduce_fused,
    matmul_all_reduce_fp4,
    matmul_all_reduce_interference,
    matmul_all_reduce_split,
    resource_report,
)
from resource_model import default_resource_table


NUM_SMS = 132
TRACE_POINTS = 4
PEAK_FRACTION_THRESHOLDS = (0.90, 0.95)


@dataclass(frozen=True)
class Config:
    mode: str
    num_comp_ctas: int
    num_comm_ctas: int
    comm_threads: int
    comm_smem_kib: int
    fp4_mode: str | None = None


def init_distributed() -> tuple[int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError("ParallelKittens GEMM+AR requires exactly 8 ranks")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    torch.manual_seed(1234 + local_rank)
    return local_rank, world_size


def unique_sm_count(smids: torch.Tensor, valid: int) -> int:
    values = smids[:valid]
    values = values[values >= 0]
    return int(torch.unique(values).numel())


def sm_role_counts(
    compute_smids: torch.Tensor,
    communication_smids: torch.Tensor,
    num_comp: int,
    num_comm: int,
) -> dict[str, int]:
    compute = {
        int(value) for value in compute_smids[:num_comp].cpu().tolist()
        if int(value) >= 0
    }
    communication = {
        int(value) for value in communication_smids[:num_comm].cpu().tolist()
        if int(value) >= 0
    }
    return {
        "compute_distinct_sms": len(compute),
        "comm_distinct_sms": len(communication),
        "overlap_sms": len(compute & communication),
        "union_sms": len(compute | communication),
    }


def allocate_diagnostics(
    device: torch.device, num_comp: int, num_comm: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    compute_trace = torch.zeros(
        (max(1, num_comp), TRACE_POINTS), dtype=torch.int64, device=device
    )
    comm_trace = torch.zeros(
        (max(1, num_comm), TRACE_POINTS), dtype=torch.int64, device=device
    )
    compute_smids = torch.full(
        (max(1, num_comp),), -1, dtype=torch.int32, device=device
    )
    comm_smids = torch.full(
        (max(1, num_comm),), -1, dtype=torch.int32, device=device
    )
    return compute_trace, comm_trace, compute_smids, comm_smids


def time_distributed_cuda_operation(
    operation: Callable[[], None],
    warmups: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iterations):
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    # Reduce the whole sample vector at once after all device collectives have
    # finished.  Interleaving an NCCL MAX after every custom NVLS operation can
    # race the next system-scope barrier reuse and deadlock some ranks.
    elapsed = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return [float(item) for item in elapsed.cpu().tolist()]


def time_communication_operation(
    reset: Callable[[], None],
    operation: Callable[[], None],
    warmups: int,
    iterations: int,
    device: torch.device,
    sample_metric: Callable[[], float] | None = None,
) -> tuple[list[float], list[float] | None]:
    for _ in range(warmups):
        reset()
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    samples: list[float] = []
    metric_samples: list[float] | None = (
        [] if sample_metric is not None else None
    )
    for _ in range(iterations):
        reset()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        if sample_metric is not None:
            assert metric_samples is not None
            metric_samples.append(sample_metric())
    elapsed = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    reduced_samples = [float(item) for item in elapsed.cpu().tolist()]
    if metric_samples is None:
        return reduced_samples, None

    metrics = torch.tensor(metric_samples, dtype=torch.float64, device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
    reduced_metrics = [float(item) for item in metrics.cpu().tolist()]
    return reduced_samples, reduced_metrics


def make_tensors(
    m: int, k: int, n: int, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, TKParallelTensor, TKParallelTensor]:
    device = torch.device("cuda", rank)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=device) / k**0.25
    b = torch.randn((k, n), dtype=torch.bfloat16, device=device) / k**0.25
    c = TKParallelTensor(
        (m, n),
        dtype=torch.bfloat16,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    barrier = TKParallelTensor(
        (2, 1024, 1024),
        dtype=torch.int32,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    c.data_.zero_()
    barrier.data_.zero_()
    dist.barrier()
    return a, b, c, barrier


def make_fp4_tensors(
    m: int, n: int, rank: int, world_size: int
) -> tuple[TKParallelTensor, TKParallelTensor]:
    data = TKParallelTensor(
        (world_size, m, n // 2),
        dtype=torch.uint8,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    scales = TKParallelTensor(
        (world_size, m, n // 16),
        dtype=torch.bfloat16,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    data.data_.zero_()
    scales.data_.zero_()
    dist.barrier()
    return data, scales


def make_parallel_output(
    m: int, n: int, rank: int, world_size: int
) -> tuple[TKParallelTensor, TKParallelTensor]:
    c = TKParallelTensor(
        (m, n),
        dtype=torch.bfloat16,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    barrier = TKParallelTensor(
        (2, 1024, 1024),
        dtype=torch.int32,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )
    c.data_.fill_(rank + 1)
    barrier.data_.zero_()
    dist.barrier()
    return c, barrier


def operation_for_config(
    config: Config,
    a: torch.Tensor,
    b: torch.Tensor,
    c: TKParallelTensor,
    barrier: TKParallelTensor,
    diagnostics: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    instrument: bool,
    fp4_data: TKParallelTensor | None = None,
    fp4_scales: TKParallelTensor | None = None,
) -> Callable[[], None]:
    compute_trace, comm_trace, compute_smids, comm_smids = diagnostics
    if config.mode == "fused":
        return lambda: matmul_all_reduce_fused(
            a, b, c, barrier, config.num_comm_ctas
        )
    if config.fp4_mode is not None:
        if fp4_data is None or fp4_scales is None:
            raise ValueError("FP4 config requires FP4 output tensors")
        return lambda: matmul_all_reduce_fp4(
            a,
            b,
            c,
            fp4_data,
            fp4_scales,
            barrier,
            config.mode,
            config.fp4_mode,
            config.num_comp_ctas,
            config.num_comm_ctas,
            config.comm_threads,
            instrument,
            compute_trace,
            comm_trace,
            compute_smids,
            comm_smids,
        )
    return lambda: matmul_all_reduce_split(
        a,
        b,
        c,
        barrier,
        config.mode,
        config.num_comp_ctas,
        config.num_comm_ctas,
        config.comm_threads,
        config.comm_smem_kib * 1024,
        instrument,
        compute_trace,
        comm_trace,
        compute_smids,
        comm_smids,
    )


def fp4_encode_nibble(value: torch.Tensor) -> torch.Tensor:
    """Encode normalized values using the software E2M1 thresholds."""
    magnitude = value.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int32)
    code = torch.where(magnitude >= 0.25, torch.ones_like(code), code)
    code = torch.where(magnitude >= 0.75, torch.full_like(code, 2), code)
    code = torch.where(magnitude >= 1.25, torch.full_like(code, 3), code)
    code = torch.where(magnitude >= 1.75, torch.full_like(code, 4), code)
    code = torch.where(magnitude >= 2.5, torch.full_like(code, 5), code)
    code = torch.where(magnitude >= 3.5, torch.full_like(code, 6), code)
    code = torch.where(magnitude >= 5.0, torch.full_like(code, 7), code)
    return code | (value.lt(0).to(torch.int32) << 3)


def fp4_quantize_bf16(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a BF16 matrix to packed E2M1 with one BF16 scale per 16."""
    if values.dim() != 2 or values.shape[-1] % 16:
        raise ValueError("FP4 reference expects a 2D matrix with N divisible by 16")
    rounded = values.to(torch.bfloat16).float()
    blocks = rounded.reshape(values.shape[0], -1, 16)
    amax = blocks.abs().amax(dim=-1)
    scales = (amax / 6.0).to(torch.bfloat16)
    scale_float = scales.float().unsqueeze(-1)
    normalized = torch.where(
        scale_float == 0,
        torch.zeros_like(blocks),
        blocks / scale_float,
    )
    nibble = fp4_encode_nibble(normalized)
    packed = nibble[..., 0::2] | (nibble[..., 1::2] << 4)
    return packed.to(torch.uint8).reshape(values.shape[0], -1), scales


def fp4_dequantize(
    packed: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    """Decode the packed E2M1 representation used by the device helper."""
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=packed.device,
    )
    packed_i = packed.to(torch.int32)
    low = packed_i & 0xF
    high = (packed_i >> 4) & 0xF

    def decode(code: torch.Tensor) -> torch.Tensor:
        sign = torch.where((code & 8) != 0, -1.0, 1.0)
        return sign * magnitudes[code & 7]

    values = torch.stack((decode(low), decode(high)), dim=-1).reshape(
        packed.shape[0], -1
    )
    scale_values = scales.float().unsqueeze(-1).expand(-1, -1, 16).reshape_as(values)
    return values * scale_values


def check_correctness(
    operation: Callable[[], None],
    a: torch.Tensor,
    b: torch.Tensor,
    c: TKParallelTensor,
    barrier: TKParallelTensor,
    rtol: float,
    atol: float,
    config: Config | None = None,
    fp4_data: TKParallelTensor | None = None,
    fp4_scales: TKParallelTensor | None = None,
) -> dict[str, float]:
    local_reference = torch.matmul(a, b)
    reference = local_reference.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    if config is not None and config.fp4_mode == "post":
        # POST preserves the ordinary BF16 GEMM+NVLS result and emits FP4 as
        # an additional epilogue.  Use the ordinary split path as the matched
        # arithmetic oracle: it has the same ThunderKittens GEMM and the same
        # multimem.ld_reduce.bf16x2 reduction.  NCCL/PyTorch BF16 reduction is
        # not bit-equivalent to that device primitive's accumulation order.
        control_diagnostics = allocate_diagnostics(
            a.device, config.num_comp_ctas, config.num_comm_ctas
        )
        matmul_all_reduce_split(
            a,
            b,
            c,
            barrier,
            "default",
            config.num_comp_ctas,
            config.num_comm_ctas,
            config.comm_threads,
            0,
            False,
            *control_diagnostics,
        )
        torch.cuda.synchronize()
        reference = c.data_.clone()
    if config is not None and config.fp4_mode == "pre":
        matmul_local_bf16(a, b, c, barrier, config.num_comp_ctas)
        torch.cuda.synchronize()
        local_reference = c.data_.clone()
        local_data, local_scales = fp4_quantize_bf16(local_reference)
        gathered_data = [torch.empty_like(local_data) for _ in range(8)]
        gathered_scales = [torch.empty_like(local_scales) for _ in range(8)]
        dist.all_gather(gathered_data, local_data)
        dist.all_gather(gathered_scales, local_scales)
        reference = torch.zeros_like(local_reference, dtype=torch.float32)
        for rank_data, rank_scales in zip(
            gathered_data, gathered_scales, strict=True
        ):
            reference += fp4_dequantize(rank_data, rank_scales)
        # The communication kernel accumulates all rank contributions in
        # FP32, then performs one final BF16 conversion before publishing C.
        # Model that single rounding point rather than comparing the BF16
        # result with an unrounded FP32 rank sum.
        reference = reference.to(torch.bfloat16)
    operation()
    torch.cuda.synchronize()
    actual = c.data_
    diff = (actual.float() - reference.float()).abs()
    stats = {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }
    if config is not None and config.fp4_mode is not None:
        if fp4_data is None or fp4_scales is None:
            raise ValueError("FP4 correctness requires FP4 output tensors")
        expected_source = local_reference if config.fp4_mode == "pre" else actual
        expected_data, expected_scales = fp4_quantize_bf16(expected_source)
        if config.fp4_mode == "post":
            # POST assigns each 128x256 output tile to exactly one of the
            # eight communication ranks (`task_id % 8`).  The owner writes
            # that tile into its wire slice and multicasts the write to all
            # GPUs; the other seven slices intentionally remain zero.  This
            # avoids an artificial eightfold replication of the reduced
            # payload while retaining a globally visible owner-sharded wire.
            expected_data_sharded = torch.zeros_like(fp4_data.data_)
            expected_scales_sharded = torch.zeros_like(fp4_scales.data_)
            row_blocks = a.shape[0] // 128
            col_blocks = b.shape[1] // 256
            super_m = 12
            super_rows = (row_blocks // super_m) * super_m
            final_rows = row_blocks - super_rows
            super_blocks = super_m * col_blocks
            for task_id in range(row_blocks * col_blocks):
                if task_id < super_rows * col_blocks:
                    row_idx = (
                        super_m * (task_id // super_blocks)
                        + task_id % super_m
                    )
                    col_idx = (task_id % super_blocks) // super_m
                else:
                    remainder_id = task_id - super_rows * col_blocks
                    row_idx = super_rows + remainder_id % final_rows
                    col_idx = remainder_id // final_rows
                owner = task_id % 8
                row_slice = slice(row_idx * 128, (row_idx + 1) * 128)
                data_slice = slice(col_idx * 128, (col_idx + 1) * 128)
                scale_slice = slice(col_idx * 16, (col_idx + 1) * 16)
                expected_data_sharded[owner, row_slice, data_slice] = (
                    expected_data[row_slice, data_slice]
                )
                expected_scales_sharded[owner, row_slice, scale_slice] = (
                    expected_scales[row_slice, scale_slice]
                )
            expected_data = expected_data_sharded
            expected_scales = expected_scales_sharded
            actual_data = fp4_data.data_
            actual_scales = fp4_scales.data_
        else:
            # PRE stores one wire slice in each rank's local allocation.  The
            # communication kernel reads those eight peer allocations
            # directly; it intentionally does not multicast eight replicas to
            # every GPU.
            rank = dist.get_rank()
            expected_data = local_data
            expected_scales = local_scales
            actual_data = fp4_data.data_[rank]
            actual_scales = fp4_scales.data_[rank]
        torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
        stats["fp4_data_mismatch"] = float(
            (actual_data != expected_data).sum().item()
        )
        stats["fp4_scale_max_abs_diff"] = float(
            (actual_scales.float() - expected_scales.float()).abs().max().item()
        )
    torch.testing.assert_close(
        actual.float(), reference.float(), rtol=rtol, atol=atol
    )
    return stats


def check_fp4_host_barrier_control(
    a: torch.Tensor,
    b: torch.Tensor,
    c: TKParallelTensor,
    barrier: TKParallelTensor,
    fp4_data: TKParallelTensor,
    fp4_scales: TKParallelTensor,
    num_comp_ctas: int,
    num_comm_ctas: int,
    comm_threads: int,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    local_reference = torch.empty_like(c.data_)
    matmul_local_bf16(a, b, c, barrier, num_comp_ctas)
    torch.cuda.synchronize()
    local_reference.copy_(c.data_)
    local_data, local_scales = fp4_quantize_bf16(local_reference)
    gathered_data = [torch.empty_like(local_data) for _ in range(8)]
    gathered_scales = [torch.empty_like(local_scales) for _ in range(8)]
    dist.all_gather(gathered_data, local_data)
    dist.all_gather(gathered_scales, local_scales)
    reference = torch.zeros_like(local_reference, dtype=torch.float32)
    for rank_data, rank_scales in zip(gathered_data, gathered_scales, strict=True):
        reference += fp4_dequantize(rank_data, rank_scales)
    reference = reference.to(torch.bfloat16)

    fp4_pre_producer_only(
        a, b, c, fp4_data, fp4_scales, barrier, num_comp_ctas
    )
    torch.cuda.synchronize()
    dist.barrier()
    local_counter = barrier.data_[0, 0, 0].clone()
    gathered_counters = [torch.empty_like(local_counter) for _ in range(8)]
    dist.all_gather(gathered_counters, local_counter)
    fp4_pre_communication_only(
        a,
        b,
        c,
        fp4_data,
        fp4_scales,
        barrier,
        num_comm_ctas,
        comm_threads,
    )
    torch.cuda.synchronize()

    rank = dist.get_rank()
    torch.testing.assert_close(fp4_data.data_[rank], local_data, rtol=0, atol=0)
    torch.testing.assert_close(fp4_scales.data_[rank], local_scales, rtol=0, atol=0)
    diff = (c.data_.float() - reference.float()).abs()
    torch.testing.assert_close(c.data_.float(), reference.float(), rtol=rtol, atol=atol)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "fp4_data_mismatch": float((fp4_data.data_[rank] != local_data).sum().item()),
        "fp4_scale_max_abs_diff": float(
            (fp4_scales.data_[rank].float() - local_scales.float()).abs().max().item()
        ),
        "tile_counter_by_target_rank": [
            int(value.item()) for value in gathered_counters
        ],
    }


def run_config(
    config: Config,
    a: torch.Tensor,
    b: torch.Tensor,
    c: TKParallelTensor,
    barrier: TKParallelTensor,
    warmups: int,
    iterations: int,
    correctness: bool,
    rtol: float,
    atol: float,
    instrument: bool,
    record_samples: bool,
    correctness_only: bool = False,
    fp4_data: TKParallelTensor | None = None,
    fp4_scales: TKParallelTensor | None = None,
) -> dict[str, object]:
    diagnostics = allocate_diagnostics(
        a.device, config.num_comp_ctas, config.num_comm_ctas
    )
    operation = operation_for_config(
        config, a, b, c, barrier, diagnostics, instrument, fp4_data, fp4_scales
    )
    correctness_stats: dict[str, float] | None = None
    if correctness:
        correctness_stats = check_correctness(
            operation,
            a,
            b,
            c,
            barrier,
            rtol,
            atol,
            config,
            fp4_data,
            fp4_scales,
        )

    if correctness_only:
        return {
            "kind": "gemm_all_reduce",
            "mode": config.mode,
            "num_comp_ctas": config.num_comp_ctas,
            "num_comm_ctas": config.num_comm_ctas,
            "comm_threads": config.comm_threads,
            "comm_smem_kib": config.comm_smem_kib,
            "fp4_mode": config.fp4_mode,
            "sample_count": 0,
            "instrumented": instrument,
            "correctness_only": True,
            "correctness": correctness_stats,
        }

    samples = time_distributed_cuda_operation(
        operation, warmups, iterations, a.device
    )
    global_median = statistics.median(samples)
    global_mean = statistics.mean(samples)

    compute_trace, comm_trace, compute_smids, comm_smids = diagnostics
    result: dict[str, object] = {
        "kind": "gemm_all_reduce",
        "mode": config.mode,
        "num_comp_ctas": config.num_comp_ctas,
        "num_comm_ctas": config.num_comm_ctas,
        "comm_threads": config.comm_threads,
        "comm_smem_kib": config.comm_smem_kib,
        "fp4_mode": config.fp4_mode,
        "max_rank_median_ms": global_median,
        "max_rank_mean_ms": global_mean,
        "max_rank_min_ms": min(samples),
        "max_rank_p95_ms": sorted(samples)[int(0.95 * (len(samples) - 1))],
        "max_rank_max_ms": max(samples),
        "sample_count": len(samples),
        "instrumented": instrument,
        "compute_distinct_sms": unique_sm_count(
            compute_smids, config.num_comp_ctas
        ) if instrument and config.mode != "fused" else None,
        "comm_distinct_sms": unique_sm_count(
            comm_smids, config.num_comm_ctas
        ) if instrument and config.mode != "fused" else None,
        "correctness": correctness_stats,
    }
    if config.mode != "fused":
        result["resources"] = resource_report(
            config.comm_threads, config.comm_smem_kib * 1024
        )
    if record_samples:
        result["samples_ms"] = samples
    if instrument and config.mode != "fused":
        result["trace_ns"] = {
            "compute": compute_trace[: config.num_comp_ctas].cpu().tolist(),
            "communication": comm_trace[: config.num_comm_ctas].cpu().tolist(),
        }
    return result


def run_communication_sweep(
    a: torch.Tensor,
    b: torch.Tensor,
    c: TKParallelTensor,
    barrier: TKParallelTensor,
    thread_values: list[int],
    cta_values: list[int],
    smem_kib: int,
    warmups: int,
    iterations: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    source = c.data_.clone()
    for threads, ctas in itertools.product(thread_values, cta_values):
        _, comm_trace, _, comm_smids = allocate_diagnostics(a.device, 1, ctas)

        def reset() -> None:
            c.data_.copy_(source)

        def operation(instrument: bool = False) -> None:
            all_reduce_split(
                a,
                b,
                c,
                barrier,
                ctas,
                threads,
                smem_kib * 1024,
                instrument,
                comm_trace,
                comm_smids,
            )

        samples, _ = time_communication_operation(
            reset, operation, warmups, iterations, a.device
        )
        local_median = statistics.median(samples)
        max_median = local_median
        payload_bytes = c.data_.numel() * c.data_.element_size()
        logical_gbps = payload_bytes * 1e-9 / (max_median * 1e-3)
        ring_equivalent_gbps = logical_gbps * 2 * (8 - 1) / 8
        results.append(
            {
                "kind": "communication_only",
                "num_comm_ctas": ctas,
                "comm_threads": threads,
                "comm_smem_kib": smem_kib,
                "max_rank_median_ms": max_median,
                "logical_payload_gbps": logical_gbps,
                "ring_equivalent_gbps": ring_equivalent_gbps,
                "comm_distinct_sms": None,
                "resources": resource_report(threads, smem_kib * 1024),
            }
        )
    annotate_peak_thresholds(results)
    return results


def annotate_peak_thresholds(results: list[dict[str, object]]) -> None:
    peak = max(float(item["logical_payload_gbps"]) for item in results)
    for item in results:
        fraction = float(item["logical_payload_gbps"]) / peak
        item["fraction_of_measured_peak"] = fraction
        for threshold in PEAK_FRACTION_THRESHOLDS:
            pct = int(round(threshold * 100))
            item[f"meets_{pct}pct_peak"] = fraction >= threshold

    for threshold in PEAK_FRACTION_THRESHOLDS:
        pct = int(round(threshold * 100))
        candidates = [
            item for item in results if bool(item[f"meets_{pct}pct_peak"])
        ]
        selected = min(
            candidates,
            key=lambda item: (
                int(item["num_comm_ctas"]),
                int(item["comm_threads"]),
            ),
        )
        for item in results:
            item[f"minimum_{pct}pct_peak_config"] = item is selected


def run_interference_sweep(
    a: torch.Tensor,
    b: torch.Tensor,
    compute_c: TKParallelTensor,
    compute_barrier: TKParallelTensor,
    communication_c: TKParallelTensor,
    communication_barrier: TKParallelTensor,
    cta_values: list[int],
    threads: int,
    smem_kib: int,
    warmups: int,
    iterations: int,
    record_samples: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    source = communication_c.data_.clone()
    for ctas in cta_values:
        num_comp = NUM_SMS - ctas
        diagnostics = allocate_diagnostics(a.device, num_comp, ctas)
        compute_trace, comm_trace, compute_smids, comm_smids = diagnostics

        def reset() -> None:
            communication_c.data_.copy_(source)

        def operation() -> None:
            matmul_all_reduce_interference(
                a,
                b,
                compute_c,
                compute_barrier,
                communication_c,
                communication_barrier,
                num_comp,
                ctas,
                threads,
                smem_kib * 1024,
                True,
                compute_trace,
                comm_trace,
                compute_smids,
                comm_smids,
            )

        def communication_span_ms() -> float:
            return (
                int(comm_trace[:ctas, 3].max().item())
                - int(comm_trace[:ctas, 0].min().item())
            ) * 1e-6

        samples, comm_span_samples = time_communication_operation(
            reset,
            operation,
            warmups,
            iterations,
            a.device,
            sample_metric=communication_span_ms,
        )
        assert comm_span_samples is not None
        torch.cuda.synchronize()
        role_counts = sm_role_counts(
            compute_smids, comm_smids, num_comp, ctas
        )
        comm_span_ms = statistics.median(comm_span_samples)
        comm_span_p95_ms = sorted(comm_span_samples)[
            int(0.95 * (len(comm_span_samples) - 1))
        ]
        payload_bytes = (
            communication_c.data_.numel()
            * communication_c.data_.element_size()
        )
        logical_gbps = payload_bytes * 1e-9 / (comm_span_ms * 1e-3)
        record: dict[str, object] = {
            "kind": "communication_under_gemm_interference",
            "num_comp_ctas": num_comp,
            "num_comm_ctas": ctas,
            "comm_threads": threads,
            "comm_smem_kib": smem_kib,
            "max_rank_median_ms": statistics.median(samples),
            "max_rank_mean_ms": statistics.mean(samples),
            "max_rank_p95_ms": sorted(samples)[
                int(0.95 * (len(samples) - 1))
            ],
            "communication_grid_span_ms": comm_span_ms,
            "communication_grid_span_p95_ms": comm_span_p95_ms,
            "logical_payload_gbps": logical_gbps,
            "resources": resource_report(threads, smem_kib * 1024),
            **role_counts,
        }
        if record_samples:
            record["samples_ms"] = samples
            record["communication_grid_span_samples_ms"] = (
                comm_span_samples
            )
        results.append(record)

    annotate_peak_thresholds(results)
    return results


def preset_configs(name: str) -> list[Config]:
    if name == "smoke":
        return [
            Config("fused", 124, 8, 384, 128),
            Config("default", 132, 8, 384, 128),
            Config("pdl_grid", 132, 8, 384, 128),
            Config("pdl_tile", 132, 8, 384, 128),
        ]
    if name == "fp4":
        return [
            Config("default", 132, 8, 1024, 0, "pre"),
            Config("pdl_grid", 132, 8, 1024, 0, "pre"),
            Config("pdl_tile", 132, 8, 1024, 0, "pre"),
            Config("default", 132, 8, 1024, 0, "post"),
            Config("pdl_grid", 132, 8, 1024, 0, "post"),
            Config("pdl_tile", 132, 8, 1024, 0, "post"),
        ]
    if name == "mechanism":
        return [
            Config(mode, 132, 8, 384, 128)
            for mode in ["default", "pdl_grid", "pdl_tile"]
        ] + [
            Config("fused", 124, 8, 384, 128),
        ]
    if name == "full":
        configs = [Config("fused", NUM_SMS - c, c, 384, 128) for c in [1, 2, 4, 8, 16, 32]]
        for mode in ["default", "pdl_grid", "pdl_tile"]:
            configs.extend(
                Config(mode, 132, c, threads, 128)
                for c in [1, 2, 4, 8, 16, 32]
                for threads in [256, 384, 1024]
            )
        configs.extend(
            Config("two_stream", NUM_SMS - c, c, threads, 128)
            for c in [1, 2, 4, 8, 16, 32]
            for threads in [256, 384, 512, 768, 1024]
        )
        return configs
    raise ValueError(name)


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--preset", choices=["smoke", "mechanism", "full", "fp4"], default="smoke")
    parser.add_argument(
        "--fp4-mode",
        choices=["pre", "post"],
        help="run only one FP4 producer mode when --preset fp4 is selected",
    )
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="run one correctness invocation and skip warmup/timed replays",
    )
    parser.add_argument(
        "--fp4-host-barrier-control",
        action="store_true",
        help="run PRE producer-only, host barrier, then communication-only",
    )
    parser.add_argument(
        "--fp4-producer-only-control",
        action="store_true",
        help="run one PRE producer and report its tile-counter footprint",
    )
    parser.add_argument(
        "--fp4-producer-control-comp-ctas",
        type=int,
        help="override compute CTA count for the PRE producer-only diagnostic",
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260809,
        help="seed used to randomize configuration order",
    )
    parser.add_argument("--comm-sweep", action="store_true")
    parser.add_argument("--comm-threads", default="256,384,512,768,1024")
    parser.add_argument("--comm-ctas", default="1,2,4,8,12,16,24,32")
    parser.add_argument("--comm-smem-kib", type=int, default=128)
    parser.add_argument("--interference-sweep", action="store_true")
    parser.add_argument(
        "--interference-ctas", default="4,8,12,16,20,24,28,32"
    )
    parser.add_argument("--interference-threads", type=int, default=1024)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--record-samples", action="store_true")
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="collect globaltimer/SMID diagnostics; disabled for latency runs",
    )
    parser.add_argument(
        "--modes",
        help="optional comma-separated subset of fused,default,pdl_grid,pdl_tile,two_stream",
    )
    parser.add_argument(
        "--mode-order",
        help="explicit comma-separated mode order; overrides preset order",
    )
    parser.add_argument(
        "--comparison-comm-ctas",
        type=int,
        help="set communication roles/CTAs for every comparison mode",
    )
    parser.add_argument(
        "--comparison-comm-threads",
        type=int,
        default=1024,
        help="independent communication CTA shape for split modes",
    )
    parser.add_argument(
        "--fused-comm-ctas",
        type=int,
        help="override persistent fused communication SM roles",
    )
    parser.add_argument(
        "--split-comm-ctas",
        type=int,
        help="override communication CTAs for default/PDL split modes",
    )
    parser.add_argument(
        "--split-comm-threads",
        type=int,
        default=1024,
        help="communication threads for default/PDL split modes",
    )
    parser.add_argument(
        "--two-stream-ctas",
        type=int,
        help="override the two-stream communication CTA count",
    )
    parser.add_argument(
        "--two-stream-threads",
        type=int,
        default=1024,
        help="communication threads for an overridden two-stream config",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.correctness_only:
        args.correctness = True

    rank, world_size = init_distributed()
    a, b, c, barrier = make_tensors(args.m, args.k, args.n, rank, world_size)
    fp4_data: TKParallelTensor | None = None
    fp4_scales: TKParallelTensor | None = None
    if args.preset == "fp4":
        fp4_data, fp4_scales = make_fp4_tensors(args.m, args.n, rank, world_size)
    records: list[dict[str, object]] = []

    if rank == 0:
        records.append(
            {
                "kind": "resource_model",
                "models": default_resource_table(),
            }
        )

    if args.comm_sweep:
        sweep = run_communication_sweep(
            a,
            b,
            c,
            barrier,
            parse_int_list(args.comm_threads),
            parse_int_list(args.comm_ctas),
            args.comm_smem_kib,
            args.warmups,
            args.iterations,
        )
        if rank == 0:
            records.extend(sweep)

    if args.interference_sweep:
        communication_c, communication_barrier = make_parallel_output(
            args.m, args.n, rank, world_size
        )
        interference = run_interference_sweep(
            a,
            b,
            c,
            barrier,
            communication_c,
            communication_barrier,
            parse_int_list(args.interference_ctas),
            args.interference_threads,
            args.comm_smem_kib,
            args.warmups,
            args.iterations,
            args.record_samples,
        )
        if rank == 0:
            records.extend(interference)

    configs = preset_configs(args.preset)
    if args.fp4_mode is not None:
        if args.preset != "fp4":
            raise ValueError("--fp4-mode requires --preset fp4")
        configs = [config for config in configs if config.fp4_mode == args.fp4_mode]
    requested_mode_sequence: list[str] | None = None
    if args.modes:
        requested_mode_sequence = args.modes.split(",")
        requested_modes = set(requested_mode_sequence)
        configs = [config for config in configs if config.mode in requested_modes]
    if args.comparison_comm_ctas is not None:
        ctas = args.comparison_comm_ctas
        if not 0 < ctas < NUM_SMS:
            raise ValueError("comparison communication CTAs must be in 1..131")
        configs = [
            Config(
                config.mode,
                NUM_SMS - ctas
                if config.mode in {"fused", "two_stream"}
                else NUM_SMS,
                ctas,
                config.comm_threads
                if config.mode == "fused"
                else args.comparison_comm_threads,
                args.comm_smem_kib,
                config.fp4_mode,
            )
            for config in configs
        ]
    if args.fused_comm_ctas is not None:
        ctas = args.fused_comm_ctas
        if not 0 < ctas < NUM_SMS:
            raise ValueError("fused communication roles must be in 1..131")
        configs = [
            Config(
                config.mode,
                NUM_SMS - ctas,
                ctas,
                config.comm_threads,
                config.comm_smem_kib,
                config.fp4_mode,
            )
            if config.mode == "fused"
            else config
            for config in configs
        ]
    if args.split_comm_ctas is not None:
        ctas = args.split_comm_ctas
        if not 0 < ctas <= NUM_SMS:
            raise ValueError("split communication CTAs must be in 1..132")
        configs = [
            Config(
                config.mode,
                NUM_SMS,
                ctas,
                args.split_comm_threads,
                args.comm_smem_kib,
                config.fp4_mode,
            )
            if config.mode in {"default", "pdl_grid", "pdl_tile"}
            else config
            for config in configs
        ]
    if args.two_stream_ctas is not None:
        configs = [config for config in configs if config.mode != "two_stream"]
        configs.append(
            Config(
                "two_stream",
                NUM_SMS - args.two_stream_ctas,
                args.two_stream_ctas,
                args.two_stream_threads,
                args.comm_smem_kib,
                None,
            )
        )
    if args.modes and "two_stream" in requested_modes and not any(
        config.mode == "two_stream" for config in configs
    ):
        raise ValueError(
            "two_stream is diagnostic-only; pass --two-stream-ctas explicitly"
        )
    if args.mode_order and args.randomize:
        raise ValueError("--mode-order and --randomize are mutually exclusive")
    if args.mode_order:
        by_mode = {config.mode: config for config in configs}
        requested_order = args.mode_order.split(",")
        missing = [mode for mode in requested_order if mode not in by_mode]
        if missing:
            raise ValueError(f"mode order contains unavailable modes: {missing}")
        configs = [by_mode[mode] for mode in requested_order]
    elif requested_mode_sequence is not None:
        by_mode = {config.mode: config for config in configs}
        configs = [by_mode[mode] for mode in requested_mode_sequence]
    if args.randomize:
        random.Random(args.random_seed).shuffle(configs)
    if args.sweep_only:
        configs = []
    if args.fp4_producer_only_control:
        if args.preset != "fp4" or args.fp4_mode != "pre":
            raise ValueError(
                "--fp4-producer-only-control requires --preset fp4 --fp4-mode pre"
            )
        assert fp4_data is not None and fp4_scales is not None
        control_config = configs[0]
        control_comp_ctas = (
            args.fp4_producer_control_comp_ctas
            if args.fp4_producer_control_comp_ctas is not None
            else control_config.num_comp_ctas
        )
        if not 0 < control_comp_ctas <= NUM_SMS:
            raise ValueError("producer control compute CTAs must be in 1..132")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fp4_pre_producer_only(
            a,
            b,
            c,
            fp4_data,
            fp4_scales,
            barrier,
            control_comp_ctas,
        )
        end.record()
        end.synchronize()
        row_blocks = args.m // 128
        col_blocks = args.n // 256
        counters = barrier.data_[0, :row_blocks, :col_blocks]
        nonzero = counters[counters != 0]
        record = {
            "kind": "fp4_producer_only_control",
            "elapsed_ms": float(start.elapsed_time(end)),
            "counter_nonzero": int(nonzero.numel()),
            "counter_min_nonzero": int(nonzero.min().item()) if nonzero.numel() else 0,
            "counter_max": int(counters.max().item()),
            "counter_sum": int(counters.sum().item()),
            "num_comp_ctas": control_comp_ctas,
        }
        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, record)
        if rank == 0:
            print(json.dumps({"per_rank": gathered, **record}))
        return
    if args.fp4_host_barrier_control:
        if args.preset != "fp4" or args.fp4_mode != "pre":
            raise ValueError(
                "--fp4-host-barrier-control requires --preset fp4 --fp4-mode pre"
            )
        assert fp4_data is not None and fp4_scales is not None
        control_config = configs[0]
        control_stats = check_fp4_host_barrier_control(
            a,
            b,
            c,
            barrier,
            fp4_data,
            fp4_scales,
            control_config.num_comp_ctas,
            control_config.num_comm_ctas,
            control_config.comm_threads,
            args.rtol,
            args.atol,
        )
        if rank == 0:
            print(json.dumps({"kind": "fp4_host_barrier_control", **control_stats}))
        return
    for order_index, config in enumerate(configs):
        # A previous mode may use a non-blocking auxiliary stream.  Start every
        # configuration from a globally quiescent point so its correctness and
        # timing are independent of randomized mode order.
        torch.cuda.synchronize()
        dist.barrier()
        result = run_config(
            config,
            a,
            b,
            c,
            barrier,
            args.warmups,
            args.iterations,
            args.correctness,
            args.rtol,
            args.atol,
            args.instrument,
            args.record_samples,
            args.correctness_only,
            fp4_data,
            fp4_scales,
        )
        result.update(
            {
                "m": args.m,
                "k": args.k,
                "n": args.n,
                "order_index": order_index,
                "random_seed": args.random_seed if args.randomize else None,
            }
        )
        if rank == 0:
            records.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)

    if rank == 0 and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
