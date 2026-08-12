from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineCost:
    local_reduce_ms: float
    inter_node_ms: float
    local_broadcast_ms: float
    control_ms: float
    num_chunks: int
    observed_sequential_ms: float | None = None
    observed_pipeline_ms: float | None = None
    flat_ms: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.local_reduce_ms,
            self.inter_node_ms,
            self.local_broadcast_ms,
            self.control_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("stage times must be non-negative")
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")

    @property
    def isolated_sequential_ms(self) -> float:
        return (
            self.local_reduce_ms
            + self.inter_node_ms
            + self.local_broadcast_ms
            + self.control_ms
        )

    @property
    def lower_bound_ms(self) -> float:
        stage_per_chunk = (
            self.local_reduce_ms / self.num_chunks,
            self.inter_node_ms / self.num_chunks,
            self.local_broadcast_ms / self.num_chunks,
        )
        fill = sum(stage_per_chunk)
        steady_state = (self.num_chunks - 1) * max(stage_per_chunk)
        return fill + steady_state + self.control_ms

    @property
    def overlap_efficiency(self) -> float | None:
        if self.observed_pipeline_ms is None:
            return None
        sequential = (
            self.observed_sequential_ms
            if self.observed_sequential_ms is not None
            else self.isolated_sequential_ms
        )
        opportunity = sequential - self.lower_bound_ms
        if opportunity <= 0:
            return None
        return (sequential - self.observed_pipeline_ms) / opportunity

    @property
    def perfect_overlap_opportunity_vs_flat(self) -> float | None:
        if self.flat_ms is None or self.flat_ms <= 0:
            return None
        return (self.flat_ms - self.lower_bound_ms) / self.flat_ms

    def as_dict(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = asdict(self)
        result.update(
            isolated_sequential_ms=self.isolated_sequential_ms,
            pipeline_lower_bound_ms=self.lower_bound_ms,
            overlap_efficiency=self.overlap_efficiency,
            perfect_overlap_opportunity_vs_flat=(
                self.perfect_overlap_opportunity_vs_flat
            ),
        )
        return result
