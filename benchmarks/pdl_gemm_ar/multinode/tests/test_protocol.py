from __future__ import annotations

import threading
import unittest

from ..loopback_backend import LoopbackTransport
from ..cost_model import PipelineCost
from ..pk_pipeline import (
    TILE_BYTES_BF16,
    OwnerWireLayout,
    PipelineConfig,
)
from ..protocol import (
    ChunkPlan,
    CompletionBoard,
    DuplicateCompletionError,
    OutOfOrderCompletionError,
    Phase,
    StaleEpochError,
    completion_token,
)
from ..topology import RankTopology
from ..transport import TransportBuffers


class TopologyTest(unittest.TestCase):
    def test_contiguous_two_node_groups(self) -> None:
        topology = RankTopology(rank=6, world_size=8, local_world_size=4)
        self.assertEqual(topology.node_rank, 1)
        self.assertEqual(topology.local_rank, 2)
        self.assertEqual(topology.local_ranks, (4, 5, 6, 7))
        self.assertEqual(topology.ranks_in_lane(2), (2, 6))
        self.assertEqual(topology.owner_rank(5), 5)


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = ChunkPlan(total_bytes=10, chunk_bytes=4, local_world_size=2)
        self.chunks = self.plan.chunks(epoch=7)
        self.board = CompletionBoard(self.plan.num_chunks, epoch=7)

    def test_partial_chunk_and_owner_rotation(self) -> None:
        self.assertEqual([chunk.size_bytes for chunk in self.chunks], [4, 4, 2])
        self.assertEqual([chunk.owner_local_rank for chunk in self.chunks], [0, 1, 0])

    def test_order_and_exactly_once(self) -> None:
        chunk = self.chunks[0]
        with self.assertRaises(OutOfOrderCompletionError):
            self.board.publish(chunk, Phase.INTER_NODE_READY)
        self.board.publish(chunk, Phase.LOCAL_READY)
        with self.assertRaises(DuplicateCompletionError):
            self.board.publish(chunk, Phase.LOCAL_READY)

    def test_stale_epoch_after_reset(self) -> None:
        old = self.chunks[0]
        self.board.reset(8)
        with self.assertRaises(StaleEpochError):
            self.board.phase(old)
        with self.assertRaises(StaleEpochError):
            self.board.reset(8)

    def test_wait_timeout_and_notification(self) -> None:
        chunk = self.chunks[0]
        with self.assertRaises(TimeoutError):
            self.board.wait(chunk, Phase.LOCAL_READY, timeout_s=0.001)

        waiter_done = threading.Event()

        def wait_for_ready() -> None:
            self.board.wait(chunk, Phase.LOCAL_READY, timeout_s=1.0)
            waiter_done.set()

        waiter = threading.Thread(target=wait_for_ready)
        waiter.start()
        self.board.publish(chunk, Phase.LOCAL_READY)
        waiter.join(timeout=1.0)
        self.assertTrue(waiter_done.is_set())

    def test_completion_token_distinguishes_epoch_chunk_and_phase(self) -> None:
        values = {
            completion_token(7, 0, Phase.LOCAL_READY),
            completion_token(7, 1, Phase.LOCAL_READY),
            completion_token(7, 0, Phase.INTER_NODE_READY),
            completion_token(8, 0, Phase.LOCAL_READY),
        }
        self.assertEqual(len(values), 4)


class LoopbackTest(unittest.TestCase):
    def test_full_pipeline_and_epoch_reuse(self) -> None:
        topology = RankTopology(rank=0, world_size=2, local_world_size=1)
        plan = ChunkPlan(total_bytes=10, chunk_bytes=4, local_world_size=1)
        source = bytearray(range(10))
        destination = bytearray(10)
        transport = LoopbackTransport()
        chunks = plan.chunks(epoch=1)
        transport.prepare(topology, TransportBuffers(source, destination), chunks)

        for chunk in chunks:
            transport.local_reduce(chunk)
            transport.inter_node_exchange(chunk)
            transport.local_broadcast(chunk)
        transport.wait(1)
        self.assertEqual(destination, source)
        self.assertTrue(all(phase == Phase.CONSUMED for phase in transport.board.snapshot()))

        transport.reset(2)
        self.assertTrue(all(phase == Phase.EMPTY for phase in transport.board.snapshot()))
        transport.close()


class CostModelTest(unittest.TestCase):
    def test_pipeline_bound_and_overlap_efficiency(self) -> None:
        model = PipelineCost(
            compute_ms=0.0,
            local_reduce_ms=4.0,
            inter_node_ms=8.0,
            local_broadcast_ms=4.0,
            ready_control_ms=0.0,
            tail_control_ms=1.0,
            window_orchestration_ms=0.0,
            num_chunks=4,
            observed_sequential_ms=17.0,
            observed_pipeline_ms=11.0,
            flat_ms=20.0,
        )
        self.assertEqual(model.isolated_sequential_ms, 17.0)
        self.assertEqual(model.communication_estimated_ideal_overlap_ms, 11.0)
        self.assertEqual(model.estimated_ideal_overlap_ms, 11.0)
        self.assertEqual(model.overlap_efficiency, 1.0)
        self.assertEqual(model.perfect_overlap_opportunity_vs_flat, 0.45)

    def test_compute_is_a_pipeline_stage_in_the_full_bound(self) -> None:
        model = PipelineCost(
            compute_ms=8.0,
            local_reduce_ms=4.0,
            inter_node_ms=4.0,
            local_broadcast_ms=4.0,
            ready_control_ms=0.0,
            tail_control_ms=0.0,
            window_orchestration_ms=0.0,
            num_chunks=4,
        )
        self.assertEqual(model.communication_estimated_ideal_overlap_ms, 6.0)
        self.assertEqual(model.estimated_ideal_overlap_ms, 11.0)

    def test_ready_control_is_a_windowed_stage(self) -> None:
        model = PipelineCost(
            compute_ms=0.0,
            ready_control_ms=8.0,
            local_reduce_ms=4.0,
            inter_node_ms=4.0,
            local_broadcast_ms=4.0,
            tail_control_ms=1.0,
            window_orchestration_ms=0.0,
            num_chunks=4,
        )
        self.assertEqual(model.communication_estimated_ideal_overlap_ms, 12.0)
        self.assertEqual(model.estimated_ideal_overlap_ms, 12.0)

    def test_window_orchestration_is_not_counted_three_times(self) -> None:
        model = PipelineCost(
            compute_ms=0.0,
            ready_control_ms=5.0,
            local_reduce_ms=4.0,
            inter_node_ms=5.0,
            local_broadcast_ms=5.0,
            tail_control_ms=0.0,
            window_orchestration_ms=1.0,
            num_chunks=4,
        )
        self.assertEqual(model.isolated_sequential_ms, 19.0)
        self.assertEqual(model.communication_estimated_ideal_overlap_ms, 8.0)


class OwnerWireLayoutTest(unittest.TestCase):
    def test_8192_shape_has_256_slots_and_16_mib_per_rank(self) -> None:
        layout = OwnerWireLayout(m=8192, n=8192, local_rank=0)
        self.assertEqual(layout.num_tiles, 2048)
        self.assertEqual(layout.num_owner_slots, 256)
        self.assertEqual(layout.wire_bytes, 16 * 1024 * 1024)
        self.assertEqual(TILE_BYTES_BF16, 64 * 1024)

    def test_uneven_owner_counts_and_final_window(self) -> None:
        first = OwnerWireLayout(m=128, n=2304, local_rank=0)
        last = OwnerWireLayout(m=128, n=2304, local_rank=7)
        self.assertEqual(first.num_tiles, 9)
        self.assertEqual(first.num_owner_slots, 2)
        self.assertEqual(last.num_owner_slots, 1)
        self.assertEqual(first.windows(3), ((0, 2),))

    def test_pipeline_config_rejects_unbounded_wait_grid(self) -> None:
        with self.assertRaises(ValueError):
            PipelineConfig(window_tiles=0)
        with self.assertRaises(ValueError):
            PipelineConfig(window_tiles=1025)


if __name__ == "__main__":
    unittest.main()
