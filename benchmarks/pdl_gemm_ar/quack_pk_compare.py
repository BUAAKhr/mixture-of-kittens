#!/usr/bin/env python3
"""Matched 8xH100 QuACK TensorSSA + ParallelKittens comparison.

QuACK writes the rank-local PRE-communication E2M1 wire directly into the VMM
allocations owned by ``TKParallelTensor``.  The ParallelKittens consumer reads
the eight peer allocations, waits on one system-scope counter per rank/tile,
dequantizes, sums in FP32, and publishes BF16 through NVLS multicast.

The important QuACK lanes differ only in dependency scheduling:

* ``quack_default``: ordinary same-stream producer completion;
* ``quack_pdl_tail``: the normal QuACK tail trigger plus tile counters;
* ``quack_pdl_early``: trigger after the first useful AB TMA group plus tile
  counters;
* ``quack_two_stream``: diagnostic non-blocking-stream overlap without PDL.

The same process can also run the existing ThunderKittens LCSC persistent
baseline and ParallelKittens BF16 / register-FP4 split paths.  Run separate
fresh processes with rotated ``--mode-order`` values for reported latency.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist

from _C import (  # type: ignore
    TKParallelTensor,
    fp4_pre_external_pdl_communication_only,
    fp4_pre_producer_only,
    fp4_reset,
    matmul_all_reduce_fp4,
    matmul_all_reduce_fused,
    matmul_all_reduce_split,
    matmul_local_bf16,
    resource_report,
)
from quack.autotuner import _gpu_warmup
from quack.epilogue.bf16_e2m1 import (
    Bf16ScaleE2M1Store,
    WirePlanes,
    quantize_bf16_e2m1_reference,
)
from quack.epilogue.frontend import gemm_epilogue
from quack.epilogue.library import identity_epi
from quack.gemm_config import GemmConfig


NUM_SMS = 132
TILE_M = 128
TILE_N = 256
TILE_K = 64


@gemm_epilogue(
    outs={
        "wire": Bf16ScaleE2M1Store(
            "wire",
            staged_writes=True,
            full_tile_staging=True,
            tma_writes=True,
            async_tma_writes=True,
        )
    }
)
def _quack_wire_epi(acc):
    return {"wire": acc}


def init_distributed() -> tuple[int, int]:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise RuntimeError("this benchmark requires exactly eight local ranks")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    torch.manual_seed(1234 + rank)
    return rank, world_size


def make_parallel_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> TKParallelTensor:
    return TKParallelTensor(
        shape,
        dtype=dtype,
        local_rank=rank,
        local_world_size=world_size,
        multicast=True,
    )


def make_tensors(args, rank: int, world_size: int):
    device = torch.device("cuda", rank)
    a = torch.randn((args.m, args.k), dtype=torch.bfloat16, device=device) / args.k**0.25
    b = torch.randn((args.k, args.n), dtype=torch.bfloat16, device=device) / args.k**0.25
    c = make_parallel_tensor((args.m, args.n), torch.bfloat16, rank, world_size)
    data = make_parallel_tensor(
        (world_size, args.m, args.n // 2), torch.uint8, rank, world_size
    )
    scales = make_parallel_tensor(
        (world_size, args.m, args.n // 16), torch.bfloat16, rank, world_size
    )
    barrier = make_parallel_tensor((2, 1024, 1024), torch.int32, rank, world_size)
    c.data_.zero_()
    data.data_.zero_()
    scales.data_.zero_()
    barrier.data_.zero_()
    dist.barrier()
    return a, b, c, data, scales, barrier


def quack_config() -> GemmConfig:
    return GemmConfig(
        tile_m=TILE_M,
        tile_n=TILE_N,
        tile_k=TILE_K,
        pingpong=False,
        cluster_m=1,
        cluster_n=1,
        cluster_k=1,
        swap_ab=False,
        max_swizzle_size=8,
        device_capacity=torch.cuda.get_device_capability()[0],
        is_dynamic_persistent=False,
        use_tma_gather=False,
    )


def make_quack_plans(a, b, wire: WirePlanes, plain_out):
    config = quack_config()
    plain = identity_epi.plan(a, b, out={"D": plain_out}, config=config)
    ordinary = _quack_wire_epi.plan(
        a,
        b,
        out={},
        config=config,
        wire=wire,
        post_init_attrs=(("use_pdl", False),),
    )
    tail = _quack_wire_epi.plan(a, b, out={}, config=config, wire=wire)
    early = _quack_wire_epi.plan(
        a,
        b,
        out={},
        config=config,
        wire=wire,
        post_init_attrs=(("pdl_launch_after_first_load", True),),
    )
    return config, plain, ordinary, tail, early


def decode_e2m1(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
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
    expanded_scales = scales.float().unsqueeze(-1).expand(-1, -1, 16)
    return values * expanded_scales.reshape_as(values)


def allocate_diagnostics(device, num_comm_ctas: int):
    compute_trace = torch.zeros((NUM_SMS, 4), dtype=torch.int64, device=device)
    comm_trace = torch.zeros((num_comm_ctas, 4), dtype=torch.int64, device=device)
    compute_smids = torch.full((NUM_SMS,), -1, dtype=torch.int32, device=device)
    comm_smids = torch.full((num_comm_ctas,), -1, dtype=torch.int32, device=device)
    return compute_trace, comm_trace, compute_smids, comm_smids


def max_rank_samples(samples: list[float], device) -> list[float]:
    values = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return [float(value) for value in values.cpu().tolist()]


def time_interleaved_operations(
    operations: dict[str, Callable[[], None]],
    modes: list[str],
    warmups: int,
    iterations: int,
    device,
    reset_fp4: Callable[[], None],
) -> dict[str, list[float]]:
    def run(mode: str, timed: bool) -> float | None:
        if mode in {"quack_producer", "pk_fp4_producer"}:
            reset_fp4()
        if not timed:
            operations[mode]()
            if mode in {"quack_producer", "pk_fp4_producer"}:
                reset_fp4()
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operations[mode]()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        if mode in {"quack_producer", "pk_fp4_producer"}:
            reset_fp4()
        return elapsed

    for round_idx in range(warmups):
        for offset in range(len(modes)):
            run(modes[(round_idx + offset) % len(modes)], timed=False)
    torch.cuda.synchronize()
    dist.barrier()

    samples = {mode: [] for mode in modes}
    for round_idx in range(iterations):
        for offset in range(len(modes)):
            mode = modes[(round_idx + offset) % len(modes)]
            elapsed = run(mode, timed=True)
            assert elapsed is not None
            samples[mode].append(elapsed)
    return {
        mode: max_rank_samples(mode_samples, device)
        for mode, mode_samples in samples.items()
    }


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    p95_idx = int(0.95 * (len(ordered) - 1))
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p95_ms": ordered[p95_idx],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": samples,
    }


def make_operations(
    args,
    a,
    b,
    c,
    data,
    scales,
    barrier,
    wire,
    plain_out,
    plain_plan,
    ordinary_plan,
    tail_plan,
    early_plan,
):
    diagnostics = allocate_diagnostics(a.device, args.comm_ctas)
    quack_out = {}

    def run_quack(plan):
        plan.run(a, b, out=quack_out, wire=wire)

    def external_consumer(enable_pdl: bool):
        fp4_pre_external_pdl_communication_only(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            args.comm_ctas,
            args.comm_threads,
            enable_pdl,
        )

    operations: dict[str, Callable[[], None]] = {
        "quack_plain": lambda: plain_plan.run(a, b, out={"D": plain_out}),
        "pk_bf16_producer": lambda: matmul_local_bf16(
            a, b, c, barrier, NUM_SMS
        ),
        "pk_fp4_producer": lambda: fp4_pre_producer_only(
            a, b, c, data, scales, barrier, NUM_SMS
        ),
        "tk_persistent": lambda: matmul_all_reduce_fused(
            a, b, c, barrier, args.fused_comm_roles
        ),
        "pk_bf16_default": lambda: matmul_all_reduce_split(
            a,
            b,
            c,
            barrier,
            "default",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            args.comm_smem_kib * 1024,
            False,
            *diagnostics,
        ),
        "pk_bf16_pdl_grid": lambda: matmul_all_reduce_split(
            a,
            b,
            c,
            barrier,
            "pdl_grid",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            args.comm_smem_kib * 1024,
            False,
            *diagnostics,
        ),
        "pk_bf16_pdl_tile": lambda: matmul_all_reduce_split(
            a,
            b,
            c,
            barrier,
            "pdl_tile",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            args.comm_smem_kib * 1024,
            False,
            *diagnostics,
        ),
        "pk_bf16_two_stream": lambda: matmul_all_reduce_split(
            a,
            b,
            c,
            barrier,
            "two_stream",
            NUM_SMS - args.comm_ctas,
            args.comm_ctas,
            args.comm_threads,
            args.comm_smem_kib * 1024,
            False,
            *diagnostics,
        ),
        "pk_fp4_default": lambda: matmul_all_reduce_fp4(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            "default",
            "pre",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            False,
            *diagnostics,
        ),
        "pk_fp4_pdl_grid": lambda: matmul_all_reduce_fp4(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            "pdl_grid",
            "pre",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            False,
            *diagnostics,
        ),
        "pk_fp4_pdl_tile": lambda: matmul_all_reduce_fp4(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            "pdl_tile",
            "pre",
            NUM_SMS,
            args.comm_ctas,
            args.comm_threads,
            False,
            *diagnostics,
        ),
        "quack_producer": lambda: run_quack(early_plan),
        "quack_default": lambda: (run_quack(ordinary_plan), external_consumer(False)),
        "quack_pdl_tail": lambda: (run_quack(tail_plan), external_consumer(True)),
        "quack_pdl_early": lambda: (run_quack(early_plan), external_consumer(True)),
    }

    communication_stream = torch.cuda.Stream(device=a.device)
    launch_ready = torch.cuda.Event()
    communication_done = torch.cuda.Event()

    def quack_two_stream():
        launch_ready.record(torch.cuda.current_stream())
        communication_stream.wait_event(launch_ready)
        run_quack(ordinary_plan)
        with torch.cuda.stream(communication_stream):
            external_consumer(False)
            communication_done.record(communication_stream)
        torch.cuda.current_stream().wait_event(communication_done)

    operations["quack_two_stream"] = quack_two_stream
    return operations


def check_quack_correctness(
    args,
    rank,
    a,
    b,
    c,
    data,
    scales,
    barrier,
    wire,
    plain_plan,
    ordinary_plan,
    tail_plan,
    early_plan,
) -> dict[str, object]:
    plain_out = torch.empty((args.m, args.n), dtype=torch.bfloat16, device=a.device)
    plain_plan.run(a, b, out={"D": plain_out})
    torch.cuda.synchronize()
    expected_data, expected_scales = quantize_bf16_e2m1_reference(plain_out.float())
    gathered_data = [torch.empty_like(expected_data) for _ in range(8)]
    gathered_scales = [torch.empty_like(expected_scales) for _ in range(8)]
    dist.all_gather(gathered_data, expected_data)
    dist.all_gather(gathered_scales, expected_scales)
    expected_c = torch.zeros_like(plain_out, dtype=torch.float32)
    for rank_data, rank_scales in zip(gathered_data, gathered_scales, strict=True):
        expected_c += decode_e2m1(rank_data, rank_scales)
    expected_c = expected_c.to(torch.bfloat16)

    tile_rows = args.m // TILE_M
    tile_cols = args.n // TILE_N
    active_counter = barrier.data_[0, :tile_rows, :tile_cols]
    active_counter.zero_()
    early_plan.run(a, b, out={}, wire=wire)
    torch.cuda.synchronize()
    counter_min = int(active_counter.min().item())
    counter_max = int(active_counter.max().item())
    counter_sum = int(active_counter.sum().item())
    if counter_min != 1 or counter_max != 1:
        raise AssertionError(
            f"rank {rank}: QuACK producer counters are not exactly one: "
            f"min={counter_min}, max={counter_max}"
        )
    fp4_pre_external_pdl_communication_only(
        a,
        b,
        c,
        data,
        scales,
        barrier,
        args.comm_ctas,
        args.comm_threads,
        False,
    )
    torch.cuda.synchronize()

    variants = {
        "quack_default": (ordinary_plan, False),
        "quack_pdl_tail": (tail_plan, True),
        "quack_pdl_early": (early_plan, True),
    }
    results = {}
    for name, (plan, enable_pdl) in variants.items():
        c.data_.zero_()
        fp4_reset(a, b, c, data, scales, barrier)
        torch.cuda.synchronize()
        dist.barrier()
        plan.run(a, b, out={}, wire=wire)
        fp4_pre_external_pdl_communication_only(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            args.comm_ctas,
            args.comm_threads,
            enable_pdl,
        )
        torch.cuda.synchronize()
        data_mismatch = int((data.data_[rank] != expected_data).sum().item())
        scale_mismatch = int((scales.data_[rank] != expected_scales).sum().item())
        max_abs = float((c.data_.float() - expected_c.float()).abs().max().item())
        mean_abs = float((c.data_.float() - expected_c.float()).abs().mean().item())
        torch.testing.assert_close(data.data_[rank], expected_data, rtol=0, atol=0)
        torch.testing.assert_close(scales.data_[rank], expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            c.data_.float(), expected_c.float(), rtol=args.rtol, atol=args.atol
        )
        results[name] = {
            "wire_data_mismatch": data_mismatch,
            "wire_scale_mismatch": scale_mismatch,
            "output_max_abs_diff": max_abs,
            "output_mean_abs_diff": mean_abs,
            "counter_after_reset_max": int(active_counter.max().item()),
        }

    c.data_.zero_()
    fp4_reset(a, b, c, data, scales, barrier)
    torch.cuda.synchronize()
    dist.barrier()
    communication_stream = torch.cuda.Stream(device=a.device)
    launch_ready = torch.cuda.Event()
    communication_done = torch.cuda.Event()
    launch_ready.record(torch.cuda.current_stream())
    communication_stream.wait_event(launch_ready)
    ordinary_plan.run(a, b, out={}, wire=wire)
    with torch.cuda.stream(communication_stream):
        fp4_pre_external_pdl_communication_only(
            a,
            b,
            c,
            data,
            scales,
            barrier,
            args.comm_ctas,
            args.comm_threads,
            False,
        )
        communication_done.record(communication_stream)
    torch.cuda.current_stream().wait_event(communication_done)
    torch.cuda.synchronize()
    torch.testing.assert_close(data.data_[rank], expected_data, rtol=0, atol=0)
    torch.testing.assert_close(scales.data_[rank], expected_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        c.data_.float(), expected_c.float(), rtol=args.rtol, atol=args.atol
    )
    results["quack_two_stream"] = {
        "wire_data_mismatch": int((data.data_[rank] != expected_data).sum().item()),
        "wire_scale_mismatch": int(
            (scales.data_[rank] != expected_scales).sum().item()
        ),
        "output_max_abs_diff": float(
            (c.data_.float() - expected_c.float()).abs().max().item()
        ),
        "output_mean_abs_diff": float(
            (c.data_.float() - expected_c.float()).abs().mean().item()
        ),
        "counter_after_reset_max": int(active_counter.max().item()),
    }
    return {
        "producer_counter_min": counter_min,
        "producer_counter_max": counter_max,
        "producer_counter_sum": counter_sum,
        "expected_counter_sum": tile_rows * tile_cols,
        "variants": results,
    }


def check_pk_bf16_equivalence(
    args,
    a,
    b,
    c,
    barrier,
) -> dict[str, object]:
    diagnostics = allocate_diagnostics(a.device, args.comm_ctas)

    def run(mode: str, num_comp_ctas: int, num_comm_ctas: int, threads: int):
        c.data_.zero_()
        barrier.data_.zero_()
        dist.barrier()
        if mode == "fused":
            matmul_all_reduce_fused(a, b, c, barrier, num_comm_ctas)
        else:
            matmul_all_reduce_split(
                a,
                b,
                c,
                barrier,
                mode,
                num_comp_ctas,
                num_comm_ctas,
                threads,
                args.comm_smem_kib * 1024,
                False,
                *diagnostics,
            )
        torch.cuda.synchronize()
        return c.data_.clone()

    outputs = {
        "fused": run("fused", NUM_SMS - args.fused_comm_roles, args.fused_comm_roles, 384),
        "default": run("default", NUM_SMS, args.comm_ctas, args.comm_threads),
        "pdl_grid": run("pdl_grid", NUM_SMS, args.comm_ctas, args.comm_threads),
        "pdl_tile": run("pdl_tile", NUM_SMS, args.comm_ctas, args.comm_threads),
    }
    reference = outputs["default"]
    stats = {}
    for name, output in outputs.items():
        diff = (output.float() - reference.float()).abs()
        stats[name] = {
            "mismatch_vs_default": int((output != reference).sum().item()),
            "max_abs_diff_vs_default": float(diff.max().item()),
            "mean_abs_diff_vs_default": float(diff.mean().item()),
        }
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    return stats


def parse_modes(args, available: dict[str, Callable[[], None]]) -> list[str]:
    if args.mode_order:
        modes = [item for item in args.mode_order.split(",") if item]
    elif args.modes:
        modes = [item for item in args.modes.split(",") if item]
    else:
        modes = [
            "tk_persistent",
            "pk_bf16_default",
            "pk_bf16_pdl_grid",
            "pk_bf16_pdl_tile",
            "pk_fp4_default",
            "pk_fp4_pdl_grid",
            "pk_fp4_pdl_tile",
            "quack_producer",
            "quack_default",
            "quack_pdl_tail",
            "quack_pdl_early",
        ]
    unknown = [mode for mode in modes if mode not in available]
    if unknown:
        raise ValueError(f"unknown modes: {unknown}; available={sorted(available)}")
    return modes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--comm-ctas", type=int, default=20)
    parser.add_argument("--comm-threads", type=int, default=1024)
    parser.add_argument("--comm-smem-kib", type=int, default=128)
    parser.add_argument("--fused-comm-roles", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--preheat-ms", type=int, default=1000)
    parser.add_argument("--modes")
    parser.add_argument("--mode-order")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--pk-equivalence-only", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.correctness_only:
        args.correctness = True
    if args.m % TILE_M or args.n % TILE_N or args.k % TILE_K:
        parser.error("M/N/K must be divisible by 128/256/64 respectively")
    if args.comm_threads not in (256, 384, 512, 768, 1024):
        parser.error("unsupported ParallelKittens communication CTA shape")
    if not 0 < args.comm_ctas <= NUM_SMS:
        parser.error("comm CTAs must be in 1..132")

    rank, world_size = init_distributed()
    a, b, c, data, scales, barrier = make_tensors(args, rank, world_size)
    plain_out = torch.empty((args.m, args.n), dtype=torch.bfloat16, device=a.device)
    wire = WirePlanes(data.data_[rank], scales.data_[rank], barrier.data_[0])
    config, plain_plan, ordinary_plan, tail_plan, early_plan = make_quack_plans(
        a, b, wire, plain_out
    )
    torch.cuda.synchronize()
    dist.barrier()

    if args.pk_equivalence_only:
        equivalence = check_pk_bf16_equivalence(args, a, b, c, barrier)
        if rank == 0:
            print(json.dumps({"kind": "pk_bf16_equivalence", "results": equivalence}, indent=2))
        dist.destroy_process_group()
        return

    correctness = None
    if args.correctness:
        correctness = check_quack_correctness(
            args,
            rank,
            a,
            b,
            c,
            data,
            scales,
            barrier,
            wire,
            plain_plan,
            ordinary_plan,
            tail_plan,
            early_plan,
        )
        torch.cuda.synchronize()
        dist.barrier()
    if args.correctness_only:
        if rank == 0:
            print(json.dumps({"kind": "correctness", "correctness": correctness}, indent=2))
        dist.destroy_process_group()
        return

    operations = make_operations(
        args,
        a,
        b,
        c,
        data,
        scales,
        barrier,
        wire,
        plain_out,
        plain_plan,
        ordinary_plan,
        tail_plan,
        early_plan,
    )
    modes = parse_modes(args, operations)
    tile_rows = args.m // TILE_M
    tile_cols = args.n // TILE_N
    def reset_fp4():
        fp4_reset(a, b, c, data, scales, barrier)

    if args.preheat_ms:
        _gpu_warmup(args.preheat_ms)
    reset_fp4()
    torch.cuda.synchronize()
    dist.barrier()
    samples = time_interleaved_operations(
        operations,
        modes,
        args.warmups,
        args.iterations,
        a.device,
        reset_fp4,
    )
    results = {mode: summarize(samples[mode]) for mode in modes}

    record = {
        "kind": "quack_parallelkittens_comparison",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "tile": {"m": TILE_M, "n": TILE_N, "k": TILE_K},
        "tile_count": tile_rows * tile_cols,
        "tile_remainder_mod_132": (tile_rows * tile_cols) % NUM_SMS,
        "comm": {
            "ctas": args.comm_ctas,
            "threads": args.comm_threads,
            "smem_kib": args.comm_smem_kib,
            "resource_no_padding": resource_report(args.comm_threads, 0),
            "resource_with_padding": resource_report(
                args.comm_threads, args.comm_smem_kib * 1024
            ),
        },
        "fused_comm_roles": args.fused_comm_roles,
        "quack_config": config.__dict__,
        "warmups": args.warmups,
        "iterations": args.iterations,
        "preheat_ms": args.preheat_ms,
        "mode_order": modes,
        "results": results,
        "correctness": correctness,
    }
    if rank == 0:
        rendered = json.dumps(record, indent=2, sort_keys=True)
        if args.quiet:
            print(
                json.dumps(
                    {
                        "comm_ctas": args.comm_ctas,
                        "mode_order": modes,
                        "medians_ms": {
                            mode: results[mode]["median_ms"] for mode in modes
                        },
                    },
                    sort_keys=True,
                )
            )
        else:
            print(rendered)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
