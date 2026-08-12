from __future__ import annotations

import threading
import unittest

from ..loopback_backend import LoopbackTransport
from ..cost_model import PipelineCost
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
            local_reduce_ms=4.0,
            inter_node_ms=8.0,
            local_broadcast_ms=4.0,
            control_ms=1.0,
            num_chunks=4,
            observed_sequential_ms=17.0,
            observed_pipeline_ms=11.0,
            flat_ms=20.0,
        )
        self.assertEqual(model.isolated_sequential_ms, 17.0)
        self.assertEqual(model.lower_bound_ms, 11.0)
        self.assertEqual(model.overlap_efficiency, 1.0)
        self.assertEqual(model.perfect_overlap_opportunity_vs_flat, 0.45)


if __name__ == "__main__":
    unittest.main()
