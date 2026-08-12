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
from .extension import load_pdl_extension
from .nccl_backend import create_nccl_groups
from .pk_benchmark import make_tensors, parse_ints, percentile
from .pk_pipeline import PipelineConfig, PkNcclHierarchicalPipeline
from .topology import RankTopology


def summarize(samples_ms: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "p95_ms": percentile(samples_ms, 0.95),
        "samples_ms": samples_ms,
    }


def timed_samples(
    operation: Callable[[], None],
    prepare: Callable[[], None],
    post_operation: Callable[[], None],
    warmups: int,
    iterations: int,
    device: torch.device,
) -> list[float]:
    def run_once() -> float:
        prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        post_operation()
        return float(start.elapsed_time(end))

    for _ in range(warmups):
        run_once()
    samples = [run_once() for _ in range(iterations)]
    reduced = torch.tensor(samples, dtype=torch.float64, device=device)
    dist.all_reduce(reduced, op=dist.ReduceOp.MAX)
    return [float(value) for value in reduced.cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--window-tiles", type=parse_ints, default=parse_ints("4"))
    parser.add_argument("--max-inflight", type=parse_ints, default=parse_ints("1"))
    parser.add_argument("--num-comp-ctas", type=int, default=132)
    parser.add_argument("--num-pack-ctas", type=int, default=20)
    parser.add_argument("--pack-threads", type=int, default=1024)
    parser.add_argument("--unpack-threads", type=int, default=1024)
    parser.add_argument("--ready-timeout-ms", type=int, default=30_000)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--backend-label", default="nccl-native")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.trial_id < 0:
        parser.error("--trial-id must be non-negative")
    if args.warmups < 0 or args.iterations <= 0:
        parser.error("--warmups must be non-negative and --iterations positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    topology = RankTopology.from_env()
    if topology.local_world_size != 8 or topology.num_nodes < 2:
        raise RuntimeError("stage decomposition requires at least two 8-GPU nodes")

    extension = load_pdl_extension()
    groups = create_nccl_groups(topology, lane_channels=max(args.max_inflight))
    a, b, output, barrier = make_tensors(
        args.m, args.k, args.n, topology, extension
    )
    records: list[dict[str, object]] = []

    for window_tiles in args.window_tiles:
        for max_inflight in args.max_inflight:
            config = PipelineConfig(
                mode="pdl_tile",
                num_comp_ctas=args.num_comp_ctas,
                num_pack_ctas=args.num_pack_ctas,
                pack_threads=args.pack_threads,
                unpack_threads=args.unpack_threads,
                window_tiles=window_tiles,
                max_inflight=max_inflight,
                ready_timeout_ms=args.ready_timeout_ms,
            )
            stage_pipeline = PkNcclHierarchicalPipeline(
                topology, groups, a, b, output, barrier, config
            )
            control_pipeline = PkNcclHierarchicalPipeline(
                topology, groups, a, b, output, barrier, config
            )
            path_pipeline = PkNcclHierarchicalPipeline(
                topology, groups, a, b, output, barrier, config
            )

            # Seed C before measuring owner pack.  Synthetic ready-control
            # state lives in a separate pipeline instance, and the full paths
            # start from their own epoch-zero buffers.
            stage_pipeline.enqueue_compute_only()
            torch.cuda.synchronize(device)
            ready_epoch = 1

            stage_specs: list[
                tuple[str, Callable[[], None], Callable[[], None], Callable[[], None]]
            ] = [
                (
                    "window_orchestration",
                    stage_pipeline.enqueue_window_orchestration_only,
                    lambda: None,
                    lambda: None,
                ),
                (
                    "compute",
                    stage_pipeline.enqueue_compute_only,
                    lambda: None,
                    lambda: None,
                ),
                (
                    "owner_pack",
                    stage_pipeline.enqueue_pack_only,
                    lambda: None,
                    stage_pipeline.check_error,
                ),
                (
                    "ready_control",
                    lambda: control_pipeline.enqueue_ready_control_only(
                        ready_epoch
                    ),
                    lambda: control_pipeline.ready.fill_(ready_epoch),
                    control_pipeline.check_error,
                ),
                (
                    "lane_network",
                    stage_pipeline.enqueue_network_only,
                    lambda: stage_pipeline.wire.fill_(topology.node_rank + 1),
                    lambda: None,
                ),
                (
                    "nvls_unpack",
                    stage_pipeline.enqueue_unpack_only,
                    lambda: None,
                    lambda: None,
                ),
                (
                    "local_join",
                    stage_pipeline.enqueue_local_join_only,
                    lambda: None,
                    stage_pipeline.check_error,
                ),
            ]

            medians: dict[str, float] = {}
            for stage, operation, prepare, post_operation in stage_specs:
                torch.cuda.synchronize(device)
                dist.barrier()
                samples = timed_samples(
                    operation,
                    prepare,
                    post_operation,
                    args.warmups,
                    args.iterations,
                    device,
                )
                summary = summarize(samples)
                medians[stage] = float(summary["median_ms"])
                records.append(
                    {
                        "kind": "pk_hierarchical_bf16_stage",
                        "backend_label": args.backend_label,
                        "trial_id": args.trial_id,
                        "stage": stage,
                        "m": args.m,
                        "k": args.k,
                        "n": args.n,
                        "topology": topology.as_dict(),
                        "pipeline": stage_pipeline.metadata(),
                        **summary,
                    }
                )

            path_specs: list[
                tuple[str, Callable[[], None], Callable[[], None]]
            ] = [
                (
                    "flat",
                    lambda: (
                        path_pipeline.enqueue_compute_only(),
                        dist.all_reduce(output.data_, op=dist.ReduceOp.SUM),
                    ),
                    lambda: None,
                ),
                (
                    "pdl_tile",
                    path_pipeline.enqueue,
                    path_pipeline.check_error,
                ),
            ]
            path_medians: dict[str, float] = {}
            for path, operation, post_operation in path_specs:
                torch.cuda.synchronize(device)
                dist.barrier()
                samples = timed_samples(
                    operation,
                    lambda: None,
                    post_operation,
                    args.warmups,
                    args.iterations,
                    device,
                )
                summary = summarize(samples)
                path_medians[path] = float(summary["median_ms"])
                records.append(
                    {
                        "kind": "pk_hierarchical_bf16_stage_path",
                        "backend_label": args.backend_label,
                        "trial_id": args.trial_id,
                        "path": path,
                        "m": args.m,
                        "k": args.k,
                        "n": args.n,
                        "topology": topology.as_dict(),
                        "pipeline": path_pipeline.metadata(),
                        **summary,
                    }
                )

            num_windows = len(path_pipeline.layout.windows(window_tiles))
            model = PipelineCost(
                compute_ms=medians["compute"],
                local_reduce_ms=medians["owner_pack"],
                inter_node_ms=medians["lane_network"],
                local_broadcast_ms=medians["nvls_unpack"],
                ready_control_ms=medians["ready_control"],
                tail_control_ms=medians["local_join"],
                window_orchestration_ms=medians["window_orchestration"],
                num_chunks=num_windows,
                observed_pipeline_ms=path_medians["pdl_tile"],
                flat_ms=path_medians["flat"],
            )
            records.append(
                {
                    "kind": "pk_hierarchical_bf16_stage_cost_model",
                    "backend_label": args.backend_label,
                    "trial_id": args.trial_id,
                    "m": args.m,
                    "k": args.k,
                    "n": args.n,
                    "topology": topology.as_dict(),
                    "pipeline": path_pipeline.metadata(),
                    "component_medians_ms": medians,
                    "path_medians_ms": path_medians,
                    **model.as_dict(),
                }
            )

    if topology.rank == 0:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        for record in records:
            print(json.dumps(record, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
