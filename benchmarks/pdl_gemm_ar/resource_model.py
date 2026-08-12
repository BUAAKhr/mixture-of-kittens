from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TileResources:
    tile_m: int
    tile_n: int
    tile_k: int
    stages: int
    consumer_warpgroups: int
    accumulator_registers_per_consumer_thread: int
    accumulator_registers_per_cta: int
    input_smem_per_stage_bytes: int
    output_smem_bytes: int
    pipeline_smem_bytes: int
    threads_per_cta: int
    producer_registers_per_cta: int
    max_registers_per_cta_for_two_ctas: int
    max_consumer_registers_for_two_ctas: int
    max_quantized_consumer_registers_for_two_ctas: int
    accepted_target_registers_per_cta: int | None
    accepted_register_headroom_per_sm: int | None


def estimate_tile_resources(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    stages: int,
    consumer_warpgroups: int = 2,
    producer_warpgroups: int = 1,
    producer_registers: int = 40,
    register_file_size: int = 65_536,
) -> TileResources:
    if tile_m % consumer_warpgroups != 0:
        raise ValueError("tile_m must divide evenly across consumer warpgroups")
    warpgroup_threads = 128
    threads_per_cta = (
        consumer_warpgroups + producer_warpgroups
    ) * warpgroup_threads
    accumulator_per_thread = (
        tile_m * tile_n // (warpgroup_threads * consumer_warpgroups)
    )
    accumulator_per_cta = tile_m * tile_n
    smem_per_stage = 2 * (tile_m * tile_k + tile_k * tile_n)
    output_smem = 2 * tile_m * tile_n
    producer_registers_per_cta = (
        producer_warpgroups * warpgroup_threads * producer_registers
    )
    max_registers_per_cta_for_two = register_file_size // 2
    max_consumer_for_two = (
        max_registers_per_cta_for_two - producer_registers_per_cta
    ) // (consumer_warpgroups * warpgroup_threads)
    # SM90 allocates registers to a warp in 256-register granules, equivalent
    # to eight registers/thread.  Round the analytical ceiling down to a legal
    # allocation target.
    max_quantized_consumer_for_two = max_consumer_for_two // 8 * 8
    accepted_target_registers_per_cta = None
    accepted_register_headroom_per_sm = None
    if (tile_m, tile_n, tile_k, stages) == (128, 256, 64, 4):
        # The pinned ParallelKittens kernel uses setmaxnreg targets of 40 for
        # the producer warpgroup and 232 for both consumer warpgroups.
        accepted_target_registers_per_cta = (
            producer_warpgroups * warpgroup_threads * 40
            + consumer_warpgroups * warpgroup_threads * 232
        )
        accepted_register_headroom_per_sm = (
            register_file_size - accepted_target_registers_per_cta
        )
    return TileResources(
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        stages=stages,
        consumer_warpgroups=consumer_warpgroups,
        accumulator_registers_per_consumer_thread=accumulator_per_thread,
        accumulator_registers_per_cta=accumulator_per_cta,
        input_smem_per_stage_bytes=smem_per_stage,
        output_smem_bytes=output_smem,
        pipeline_smem_bytes=smem_per_stage * (stages - 1) + output_smem,
        threads_per_cta=threads_per_cta,
        producer_registers_per_cta=producer_registers_per_cta,
        max_registers_per_cta_for_two_ctas=max_registers_per_cta_for_two,
        max_consumer_registers_for_two_ctas=max_consumer_for_two,
        max_quantized_consumer_registers_for_two_ctas=(
            max_quantized_consumer_for_two
        ),
        accepted_target_registers_per_cta=accepted_target_registers_per_cta,
        accepted_register_headroom_per_sm=(
            accepted_register_headroom_per_sm
        ),
    )


def default_resource_table() -> list[dict[str, int]]:
    shapes = [
        (128, 256, 64, 4),
        (128, 128, 64, 3),
        (64, 256, 64, 2),
        (64, 128, 64, 2),
        (256, 256, 64, 2),
        (128, 512, 64, 2),
    ]
    return [asdict(estimate_tile_resources(*shape)) for shape in shapes]


if __name__ == "__main__":
    print(json.dumps(default_resource_table(), indent=2))
