from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from .extension import load_pdl_extension
from .nccl_backend import create_nccl_groups
from .pk_pipeline import PipelineConfig, PkNcclHierarchicalPipeline
from .topology import RankTopology


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item for item in value.split(",") if item)
    allowed = {"local", "flat", "default", "pdl_grid", "pdl_tile"}
    if not modes or any(mode not in allowed for mode in modes):
        raise argparse.ArgumentTypeError(
            "modes must be a subset of local,flat,default,pdl_grid,pdl_tile"
        )
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("modes must not contain duplicates")
    return modes


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def summarize(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "samples_ms": samples_ms,
    }


def make_tensors(
    m: int,
    k: int,
    n: int,
    topology: RankTopology,
    extension: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any, Any]:
    device = torch.device("cuda", topology.local_rank)
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + topology.rank)
    a = torch.randn(
        (m, k), dtype=torch.bfloat16, device=device, generator=generator
    ) / k**0.25
    b = torch.randn(
        (k, n), dtype=torch.bfloat16, device=device, generator=generator
    ) / k**0.25
    output = extension.TKParallelTensor(
        (m, n),
        dtype=torch.bfloat16,
        local_rank=topology.local_rank,
        local_world_size=topology.local_world_size,
        multicast=True,
    )
    barrier = extension.TKParallelTensor(
        (2, 1024, 1024),
        dtype=torch.int32,
        local_rank=topology.local_rank,
        local_world_size=topology.local_world_size,
        multicast=True,
    )
    output.data_.zero_()
    barrier.data_.zero_()
    dist.barrier()
    return a, b, output, barrier


def make_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    reference = torch.matmul(a, b)
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    return reference


def correctness_stats(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    difference = (actual.float() - reference.float()).abs()
    max_difference = difference.max().to(torch.float64)
    sum_difference = difference.sum().to(torch.float64)
    element_count = torch.tensor(
        [difference.numel()], dtype=torch.float64, device=difference.device
    )
    dist.all_reduce(max_difference, op=dist.ReduceOp.MAX)
    dist.all_reduce(sum_difference, op=dist.ReduceOp.SUM)
    dist.all_reduce(element_count, op=dist.ReduceOp.SUM)
    return {
        "max_abs_diff": float(max_difference.item()),
        "mean_abs_diff": float((sum_difference / element_count).item()),
    }


def timed_samples(
    operation: Callable[[], None],
    post_operation: Callable[[], None],
    warmups: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    for _ in range(warmups):
        operation()
        torch.cuda.synchronize(device)
        post_operation()

    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        post_operation()
        samples.append(float(start.elapsed_time(end)))

    reduced = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return [float(value) for value in reduced.cpu().tolist()]


def environment_record(
    args: argparse.Namespace,
    topology: RankTopology,
) -> dict[str, object]:
    return {
        "kind": "hierarchical_environment",
        "backend_label": args.backend_label,
        "trial_id": args.trial_id,
        "topology": topology.as_dict(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(topology.local_rank),
        "nccl_version": torch.cuda.nccl.version(),
        "torch_nccl_blocking_wait": os.environ.get("TORCH_NCCL_BLOCKING_WAIT"),
        "nccl_net": os.environ.get("NCCL_NET"),
        "nccl_net_plugin": os.environ.get("NCCL_NET_PLUGIN"),
        "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
        "nccl_ib_hca": os.environ.get("NCCL_IB_HCA"),
        "nccl_ib_gid_index": os.environ.get("NCCL_IB_GID_INDEX"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes("flat,default,pdl_grid,pdl_tile"),
    )
    parser.add_argument("--window-tiles", type=parse_ints, default=parse_ints("1,2,4,8,16"))
    parser.add_argument("--max-inflight", type=parse_ints, default=parse_ints("1,2,4,8"))
    parser.add_argument("--num-comp-ctas", type=int, default=132)
    parser.add_argument("--num-pack-ctas", type=int, default=20)
    parser.add_argument("--pack-threads", type=int, default=1024)
    parser.add_argument("--unpack-threads", type=int, default=1024)
    parser.add_argument("--ready-timeout-ms", type=int, default=30_000)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument(
        "--instrument",
        action="store_true",
        help="record one device-side stage timeline; keep separate from latency runs",
    )
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--randomize", action="store_true")
    parser.add_argument("--random-seed", type=int, default=20260812)
    parser.add_argument("--backend-label", default="nccl-native")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trial_id < 0:
        parser.error("--trial-id must be non-negative")
    if args.warmups < 0 or args.iterations <= 0:
        parser.error("--warmups must be non-negative and --iterations positive")
    if args.instrument and not args.correctness_only:
        parser.error(
            "--instrument requires --correctness-only; instrumented runs do "
            "not report latency"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    topology = RankTopology.from_env()
    if topology.local_world_size != 8:
        raise RuntimeError("the PK/NVLS data path requires eight GPUs per node")
    if topology.num_nodes < 2:
        raise RuntimeError("use at least two eight-GPU nodes for this benchmark")

    extension = load_pdl_extension()
    groups = create_nccl_groups(topology, lane_channels=max(args.max_inflight))
    a, b, output, barrier = make_tensors(
        args.m, args.k, args.n, topology, extension
    )
    reference = make_reference(a, b)
    device = a.device
    records: list[dict[str, object]] = []
    if topology.rank == 0:
        records.append(environment_record(args, topology))

    configurations: list[tuple[str, int | None, int | None]] = []
    if "local" in args.modes:
        configurations.append(("local", None, None))
    if "flat" in args.modes:
        configurations.append(("flat", None, None))
    for mode in args.modes:
        if mode in {"local", "flat"}:
            continue
        configurations.extend(
            (mode, window_tiles, max_inflight)
            for window_tiles in args.window_tiles
            for max_inflight in args.max_inflight
        )
    if args.randomize:
        random.Random(args.random_seed).shuffle(configurations)

    for order_index, (mode, window_tiles, max_inflight) in enumerate(configurations):
        torch.cuda.synchronize(device)
        dist.barrier()
        pipeline = None
        if mode in {"local", "flat"}:
            if mode == "local":
                operation = lambda: extension.matmul_local_bf16(
                    a, b, output, barrier, args.num_comp_ctas
                )
            else:
                operation = lambda: (
                    extension.matmul_local_bf16(
                        a, b, output, barrier, args.num_comp_ctas
                    ),
                    dist.all_reduce(output.data_, op=dist.ReduceOp.SUM),
                )
            post_operation = lambda: None
            metadata: dict[str, object] = {}
        else:
            assert window_tiles is not None and max_inflight is not None
            config = PipelineConfig(
                mode=mode,
                num_comp_ctas=args.num_comp_ctas,
                num_pack_ctas=args.num_pack_ctas,
                pack_threads=args.pack_threads,
                unpack_threads=args.unpack_threads,
                window_tiles=window_tiles,
                max_inflight=max_inflight,
                ready_timeout_ms=args.ready_timeout_ms,
                instrument=args.instrument,
            )
            pipeline = PkNcclHierarchicalPipeline(
                topology, groups, a, b, output, barrier, config
            )
            pipeline.reset_errors()
            operation = pipeline.enqueue
            post_operation = pipeline.check_error
            metadata = pipeline.metadata()

        operation()
        torch.cuda.synchronize(device)
        post_operation()
        expected = torch.matmul(a, b) if mode == "local" else reference
        correctness = correctness_stats(output.data_, expected)
        local_correct = torch.tensor(
            [int(torch.allclose(output.data_, expected, rtol=args.rtol, atol=args.atol))],
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(local_correct, op=dist.ReduceOp.MIN)
        if not bool(local_correct.item()):
            raise AssertionError(
                f"{mode} failed correctness: {json.dumps(correctness)}"
            )

        record: dict[str, object] = {
            "kind": "pk_hierarchical_bf16",
            "backend_label": args.backend_label,
            "trial_id": args.trial_id,
            "mode": mode,
            "m": args.m,
            "k": args.k,
            "n": args.n,
            "order_index": order_index,
            "random_seed": args.random_seed if args.randomize else None,
            "topology": topology.as_dict(),
            "correctness": correctness,
            "pipeline": metadata,
        }
        if pipeline is not None and args.instrument:
            record["trace"] = pipeline.trace()
        if not args.correctness_only:
            samples = timed_samples(
                operation,
                post_operation,
                args.warmups,
                args.iterations,
                device,
            )
            record.update(summarize(samples))
        if topology.rank == 0:
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    if topology.rank == 0 and args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
