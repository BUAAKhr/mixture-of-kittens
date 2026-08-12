from __future__ import annotations

import argparse
import json
import os
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist

from .cost_model import PipelineCost
from .nccl_backend import NcclHierarchicalCollective, create_nccl_groups
from .protocol import ChunkPlan
from .topology import RankTopology


def parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item)
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "samples_ms": samples_ms,
    }


def timed_samples(
    operation: Callable[[], None],
    source: torch.Tensor,
    work: torch.Tensor,
    warmups: int,
    iterations: int,
) -> list[float]:
    def run_once() -> float:
        work.copy_(source)
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(
            [float(start.elapsed_time(end))], dtype=torch.float64, device=work.device
        )
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        return float(elapsed.item())

    for _ in range(warmups):
        run_once()
    return [run_once() for _ in range(iterations)]


def validate(
    collective: NcclHierarchicalCollective,
    source: torch.Tensor,
    work: torch.Tensor,
    chunks,
    depths: tuple[int, ...],
) -> None:
    expected = source.clone()
    dist.all_reduce(expected, op=dist.ReduceOp.SUM)

    operations = {"sequential": lambda: collective.sequential(work, chunks)}
    operations.update(
        {f"window_d{depth}": lambda d=depth: collective.windowed(work, chunks, d) for depth in depths}
    )
    for name, operation in operations.items():
        work.copy_(source)
        operation()
        torch.cuda.synchronize()
        if not torch.equal(work, expected):
            difference = (work.float() - expected.float()).abs()
            raise AssertionError(
                f"{name} mismatch: max={difference.max().item()} "
                f"mean={difference.mean().item()}"
            )


def make_record(
    mode: str,
    chunk_bytes: int,
    depth: int | None,
    samples: list[float],
    topology: RankTopology,
    total_bytes: int,
    emulated_nodes: bool,
) -> dict[str, object]:
    return {
        "kind": "hierarchical_bf16_nccl",
        "mode": mode,
        "backend": "nccl",
        "dtype": "bf16",
        "chunk_bytes": chunk_bytes,
        "window_depth": depth,
        "total_bytes": total_bytes,
        "emulated_nodes": emulated_nodes,
        "topology": topology.as_dict(),
        **summarize(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=8192 * 8192)
    parser.add_argument("--logical-local-world-size", type=int)
    parser.add_argument(
        "--chunk-kib", type=parse_ints, default=parse_ints("64,128,256,512,1024")
    )
    parser.add_argument("--depths", type=parse_ints, default=parse_ints("1,2,4,8"))
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    topology = RankTopology.from_env(args.logical_local_world_size)
    groups = create_nccl_groups(topology)
    collective = NcclHierarchicalCollective(topology, groups)
    device = torch.device("cuda", local_rank)

    source = torch.full(
        (args.elements,), topology.rank + 1, dtype=torch.bfloat16, device=device
    )
    work = torch.empty_like(source)
    total_bytes = source.numel() * source.element_size()
    physical_local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    emulated_nodes = topology.local_world_size != physical_local_world_size
    records: list[dict[str, object]] = []

    for chunk_kib in args.chunk_kib:
        chunk_bytes = chunk_kib * 1024
        if chunk_bytes % source.element_size():
            raise ValueError("chunk size must align to BF16")
        plan = ChunkPlan(total_bytes, chunk_bytes, topology.local_world_size)
        chunks = plan.chunks(epoch=1)
        validate(collective, source, work, chunks, args.depths)
        if args.correctness_only:
            continue

        operations: list[tuple[str, int | None, Callable[[], None]]] = [
            ("flat", None, lambda: collective.flat_all_reduce(work)),
            ("local_reduce", None, lambda: collective.local_reduce_only(work, chunks)),
            ("inter_node", None, lambda: collective.inter_node_only(work, chunks)),
            ("local_broadcast", None, lambda: collective.local_broadcast_only(work, chunks)),
            ("sequential", None, lambda: collective.sequential(work, chunks)),
        ]
        operations.extend(
            (
                "windowed",
                depth,
                lambda d=depth: collective.windowed(work, chunks, d),
            )
            for depth in args.depths
        )

        chunk_records: list[dict[str, object]] = []
        for mode, depth, operation in operations:
            samples = timed_samples(
                operation, source, work, args.warmups, args.iterations
            )
            record = make_record(
                mode,
                chunk_bytes,
                depth,
                samples,
                topology,
                total_bytes,
                emulated_nodes,
            )
            chunk_records.append(record)
            records.append(record)

        median_by_mode = {
            (str(record["mode"]), record["window_depth"]): float(record["median_ms"])
            for record in chunk_records
        }
        best_depth = min(args.depths, key=lambda depth: median_by_mode[("windowed", depth)])
        model = PipelineCost(
            compute_ms=0.0,
            local_reduce_ms=median_by_mode[("local_reduce", None)],
            inter_node_ms=median_by_mode[("inter_node", None)],
            local_broadcast_ms=median_by_mode[("local_broadcast", None)],
            ready_control_ms=0.0,
            tail_control_ms=0.0,
            window_orchestration_ms=0.0,
            num_chunks=plan.num_chunks,
            observed_sequential_ms=median_by_mode[("sequential", None)],
            observed_pipeline_ms=median_by_mode[("windowed", best_depth)],
            flat_ms=median_by_mode[("flat", None)],
        )
        records.append(
            {
                "kind": "hierarchical_bf16_cost_model",
                "chunk_bytes": chunk_bytes,
                "best_window_depth": best_depth,
                "emulated_nodes": emulated_nodes,
                "topology": topology.as_dict(),
                **model.as_dict(),
            }
        )

    if topology.rank == 0:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps({"records": records}, sort_keys=True))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
