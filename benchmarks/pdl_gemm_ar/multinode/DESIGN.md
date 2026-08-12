# Hierarchical BF16 GEMM + AllReduce design

## Decision

The first implementation keeps four ownership layers separate:

1. ThunderKittens/ParallelKittens produces BF16 GEMM tiles and publishes the
   existing node-local tile counters.
2. A ParallelKittens owner-pack kernel performs NVLS `multimem.ld_reduce` and
   stores each owner GPU's tiles in a contiguous BF16 wire buffer.
3. NCCL exchanges the owner buffer across nodes.  The same call site can use
   the UCCL NCCL net plugin without changing the wire or PK kernels.
4. A ParallelKittens unpack kernel multicasts the completed owner tiles to all
   eight GPUs in the local NVSwitch domain.

PDL applies only to step 1 -> step 2 admission on one GPU.  Cross-node
completion is owned by NCCL/UCCL and is converted into local stream ordering
before unpack.

After unpack, each rank advances its own system-scope GPU sequence slot on
every physical copy of the multicast barrier allocation.  The path acquires
all eight source-specific slots before completion, so it directly imports each
rank's preceding multicast writes.  The sequence lives in the shared barrier,
not in a Python pipeline object, and therefore remains monotonic across stage
controls and configuration changes.  An unrelated NCCL control collective is
not used as a proxy for NVLS multicast visibility.

The current PyTorch adapter assumes that `ProcessGroupNCCL` submits on the
active CUDA stream and that `Work.wait()` inserts a GPU completion dependency
without synchronizing the CPU.  That is a version-sensitive runtime contract,
not a static guarantee made by this benchmark.  The first approved correctness
and timeline run must verify that unpack never starts before the corresponding
lane collective completes.  Its instrumentation records
`unpack_after_remote_by_window` and also requires `join_after_all_unpacks` for
the node-local GPU join.  Those CUDA-event fields prove the submitted stream
order but are not sufficient evidence that `Work.wait()` represents the real
NCCL kernel completion.  The acceptance gate also inspects the NCCL kernel end
and unpack start in an Nsight Systems timeline.  If the contract fails, the
adapter must switch to
an explicit completion event or another supported ProcessGroupNCCL primitive
before latency results are accepted.

## Data layout

The current GEMM tile is `128 x 256` BF16, exactly 64 KiB.  Global tile id
`t` is owned by local rank `t % 8`.  Each local rank packs its tiles into slot
`t // 8`, producing one contiguous wire tensor per rank.

For `M=N=8192`:

- global output: 2048 tiles, 128 MiB;
- each GPU owns 256 tiles, 16 MiB wire;
- a two-node lane AllReduce exchanges one 16 MiB owner wire per GPU;
- all eight lanes together still represent 128 MiB per node.

The hierarchy does not reduce the mathematical communication volume by itself.
Its opportunity is that NVLS handles the node-local reduction/broadcast and the
slower network only carries owner shards, with the stages overlapped against the
GEMM tail.

`window_tiles` groups adjacent owner slots into one network operation.  The
sweep `1,2,4,8,16` therefore tests 64 KiB through 1 MiB network messages.
`max_inflight` independently controls the number of CUDA streams carrying such
windows.

## Synchronization

Each owner slot has a 31-bit monotonically increasing epoch:

```text
GEMM tile publication
  -> NVLS acquire reduction
  -> owner wire stores
  -> system fence
  -> CAS ready[slot]: epoch-1 -> epoch
  -> bounded GPU wait on network stream
  -> NCCL/UCCL completion dependency
  -> NVLS release multicast
```

Duplicate/stale publication and timeout write a device error code.  The ready
slot is still released so the pipeline drains; the host checks the error after
the timed operation and raises instead of leaving ranks in an infinite spin.

The `default` producer waits for the node-local GEMM grid.  `pdl_grid` admits
the owner-pack grid early but uses a whole-grid wait.  `pdl_tile` admits early
and waits on the existing per-tile counters.  Comparing these three modes
isolates whether fine-grained producer readiness is useful once the consumer is
an actual network pipeline.

The network-stream anchor is recorded before the producer launch.  Recording
it after `hierarchical_bf16_producer` would serialize every network stream
behind compute, owner pack, and counter reset, eliminating the overlap being
tested.  The instrumentation field is therefore named `producer_path_done_ms`:
it is the completion of that complete same-stream path, not a pure GEMM event.

## NCCL and UCCL boundary

The committed NCCL reference has two roles:

- `multinode/benchmark.py` validates generic hierarchy semantics.  Its
  `windowed` mode is only batched async submission and is not claimed as tile
  overlap.
- `multinode/pk_benchmark.py` is the real PK/NVLS owner-wire experiment.  It
  uses a separate network stream, bounded GPU ready waits, lane NCCL
  collectives, and PK unpack.

The UCCL plugin launcher assigns the audited absolute
`libnccl-net-uccl.so` path to `NCCL_NET_PLUGIN`, matching the UCCL repository's
documented examples.  Preflight hashes that selected binary.  All tensor,
chunk, stream, and correctness code remains identical.

The later direct-UCCL backend must implement the following stable contract:

```text
register_region(local_wire) -> local key
exchange_remote_region(address, rkey, length)
submit_write(first_slot, slot_count, epoch) -> transfer id
poll_completion(transfer id, timeout)
publish_gpu_ready(first_slot, slot_count, epoch)
close()
```

The typed version of this contract is in `direct_transport.py`.  The first
direct backend is restricted to two nodes: each lane writes into a separate
peer receive buffer and the receiver performs a local BF16 add before unpack.
This keeps simultaneous writes from corrupting an in-place reduction buffer.
Four or more nodes require a separate ring/tree implementation and acceptance
matrix; the two-node pairwise path is not generalized implicitly.

The implementation should reuse UCCL's DMA-BUF/nvidia-peermem registration,
RDMA WRITE or WRITE_WITH_IMM, CQ polling, and explicit error propagation.  RDMA
verbs do not enter the WGMMA loop or a QuACK epilogue.

## Experimental gates

The code is prepared before execution.  No command, compile, test, preflight,
SSH session, or benchmark is authorized until the user explicitly approves
execution.  After that authorization, `run_2node.sh` is still preflight-only by
default; `compare_preflight.py` must accept both node snapshots, and
`RUN_BENCHMARK=1` is required after they are reviewed.  A real run additionally
requires `PREFLIGHT_APPROVAL_FILE`; this second, machine-generated approval
binds the backend, host set, Git SHA, submodules, worktree cleanliness, NIC
variables, and absence of stale KittensBroker files.  Preflight output defaults
to `/tmp`, so creating it does not make the audited worktree dirty.
It also binds Python/PyTorch/CUDA/driver versions, GPU limits, the complete
communication environment, the unique `_C*.so` SHA256, and the UCCL-plugin
SHA256 when selected.  Approval expires after one hour by default.
The fixed-name `KittensBroker` supports one TKParallelTensor job per node, so
the launcher also holds a node-level `flock`.
`NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, and `NCCL_IB_GID_INDEX` are mandatory on
both nodes.  UCCL-plugin approval additionally requires `NCCL_NET_PLUGIN` to be
the same existing absolute binary on both hosts.

Run order after explicit user approval:

1. Static Python/unit and shell-syntax checks on the prepared source.
2. CUDA extension build and PTXAS acceptance, including register/spill output.
3. Local protocol/correctness smoke, with no performance claim.
4. Two-node preflight and identical-SHA/binary/environment audit.
5. Review both snapshots and generate the one-hour machine approval JSON.
6. `2x8` correctness only at one 1 MiB window and one stream.
7. Separate timeline validation of the ProcessGroupNCCL `Work.wait()` stream
   contract, owner-window order, and the post-unpack eight-slot GPU join.
8. Native NCCL flat versus hierarchical producer modes.
9. `default/pdl_grid/pdl_tile` sweep over window size and in-flight depth.
10. Isolated stage decomposition for GEMM, owner pack, ready control, lane
   network, NVLS unpack, and the node-local completion join.
11. Repeat the exact matrix with the UCCL NCCL plugin.
12. Only if BF16 shows material opportunity, implement the direct UCCL adapter.

Every latency result uses three fresh processes, rotated configuration order,
20 warmups, 100 iterations, and cross-rank MAX samples.  Instrumented timeline
runs are separate from latency runs.
Each launcher invocation is one fresh-process trial and carries an explicit
`TRIAL_ID`.  Only global rank 0 writes
`<phase>_trial<id>_global_rank0.jsonl`; node 1 intentionally creates no result
file.  The three accepted latency trials use `TRIAL_ID=0,1,2` and distinct
torchrun invocations.

The direct-UCCL stage is justified only if the measured perfect-overlap bound
is at least 8% versus flat NCCL.  Productization into public PK/TK interfaces
requires at least 5% end-to-end improvement on representative `2x8` workloads
with consistent direction across three processes.  Otherwise the work remains
a research benchmark.

`PipelineCost` reports two analytical ideal-overlap estimates.  The
communication-only estimate
pipelines GPU ready control, owner pack, network, and unpack, then adds the
node-local completion join as a tail cost.  The full estimate also treats GEMM as
a stage.  An empty window fan-out/fan-in measurement is subtracted from ready,
network, and unpack before being added back once, avoiding obvious triple
counting of stream orchestration.  Both estimates still assume uniform
per-window work and ideal order alignment; neither is a formal hardware bound
or reported as measured latency.  The observed PDL path and flat NCCL
path are stored beside the model so the assumptions can be falsified.

## Complexity accounting

Each result report records:

- latency, p95, first/last remote window, achieved overlap, and compute slowdown;
- window bytes, in-flight streams, pack/unpack CTA shapes, and wire allocation;
- CPU progress use and network backend;
- changed lines, new synchronization states, streams, progress threads, and
  required environment variables.

The required ablations are flat NCCL, generic NCCL hierarchy, owner-wire
sequential, owner-wire with default readiness, whole-grid PDL, tile PDL, native
NCCL network, and UCCL plugin network.  This separates benefits due to topology,
wire layout, scheduling, and transport backend before fused compression is
introduced.
