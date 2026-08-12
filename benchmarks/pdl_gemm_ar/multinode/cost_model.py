from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PipelineCost:
    compute_ms: float
    local_reduce_ms: float
    inter_node_ms: float
    local_broadcast_ms: float
    ready_control_ms: float
    tail_control_ms: float
    window_orchestration_ms: float
    num_chunks: int
    observed_sequential_ms: float | None = None
    observed_pipeline_ms: float | None = None
    flat_ms: float | None = None

    def __post_init__(self) -> None:
        values = (
            self.compute_ms,
            self.local_reduce_ms,
            self.inter_node_ms,
            self.local_broadcast_ms,
            self.ready_control_ms,
            self.tail_control_ms,
            self.window_orchestration_ms,
        )
        if any(value < 0 for value in values):
            raise ValueError("stage times must be non-negative")
        if self.num_chunks <= 0:
            raise ValueError("num_chunks must be positive")

    @property
    def isolated_sequential_ms(self) -> float:
        return (
            self.compute_ms
            + self.local_reduce_ms
            + self.inter_node_ms
            + self.local_broadcast_ms
            + self.ready_control_ms
            + self.tail_control_ms
        )

    def _stage_without_orchestration(self, measured_ms: float) -> float:
        return max(0.0, measured_ms - self.window_orchestration_ms)

    @property
    def communication_estimated_ideal_overlap_ms(self) -> float:
        stage_per_chunk = (
            self._stage_without_orchestration(self.ready_control_ms)
            / self.num_chunks,
            self.local_reduce_ms / self.num_chunks,
            self._stage_without_orchestration(self.inter_node_ms)
            / self.num_chunks,
            self._stage_without_orchestration(self.local_broadcast_ms)
            / self.num_chunks,
        )
        fill = sum(stage_per_chunk)
        steady_state = (self.num_chunks - 1) * max(stage_per_chunk)
        return (
            fill
            + steady_state
            + self.window_orchestration_ms
            + self.tail_control_ms
        )

    @property
    def estimated_ideal_overlap_ms(self) -> float:
        # This is an optimistic uniform-window estimate.  It assumes each
        # stage's measured aggregate time is evenly divisible across windows
        # and producer tile order is perfectly aligned with owner-wire order.
        stage_per_chunk = (
            self.compute_ms / self.num_chunks,
            self._stage_without_orchestration(self.ready_control_ms)
            / self.num_chunks,
            self.local_reduce_ms / self.num_chunks,
            self._stage_without_orchestration(self.inter_node_ms)
            / self.num_chunks,
            self._stage_without_orchestration(self.local_broadcast_ms)
            / self.num_chunks,
        )
        fill = sum(stage_per_chunk)
        steady_state = (self.num_chunks - 1) * max(stage_per_chunk)
        return (
            fill
            + steady_state
            + self.window_orchestration_ms
            + self.tail_control_ms
        )

    @property
    def overlap_efficiency(self) -> float | None:
        if self.observed_pipeline_ms is None:
            return None
        sequential = (
            self.observed_sequential_ms
            if self.observed_sequential_ms is not None
            else self.isolated_sequential_ms
        )
        opportunity = sequential - self.estimated_ideal_overlap_ms
        if opportunity <= 0:
            return None
        return (sequential - self.observed_pipeline_ms) / opportunity

    @property
    def perfect_overlap_opportunity_vs_flat(self) -> float | None:
        if self.flat_ms is None or self.flat_ms <= 0:
            return None
        return (self.flat_ms - self.estimated_ideal_overlap_ms) / self.flat_ms

    def as_dict(self) -> dict[str, float | int | None]:
        result: dict[str, float | int | None] = asdict(self)
        result.update(
            isolated_sequential_ms=self.isolated_sequential_ms,
            communication_estimated_ideal_overlap_ms=(
                self.communication_estimated_ideal_overlap_ms
            ),
            estimated_ideal_overlap_ms=self.estimated_ideal_overlap_ms,
            overlap_efficiency=self.overlap_efficiency,
            perfect_overlap_opportunity_vs_flat=(
                self.perfect_overlap_opportunity_vs_flat
            ),
        )
        return result
