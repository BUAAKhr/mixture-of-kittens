# ParallelKittens GEMM+AllReduce scheduling comparison

The node-local/node-global co-design and its execution gates are specified in
[`multinode/DESIGN.md`](multinode/DESIGN.md).  The implementation is BF16-only
until that hierarchy demonstrates a measured end-to-end benefit.

The follow-on hierarchical BF16 work lives in [`multinode/`](multinode/).  Its
transport-neutral protocol deliberately separates same-node NVLS scheduling
from inter-node completion.  Start with the dependency-free protocol tests:

```bash
python -m unittest discover -s benchmarks/pdl_gemm_ar/multinode/tests -t . -v
```

The loopback backend validates chunk ownership, partial chunks, exactly-once
phase publication, epoch reuse, and timeout behavior.  It is not a performance
model for NVLink or RDMA.

The NCCL reference benchmark can emulate two logical four-GPU nodes on one
eight-GPU host.  This validates communicator ordering and hierarchical BF16
semantics, but all traffic still uses the host's NVLink/NVSwitch fabric.  Its
`windowed` mode batches asynchronous submissions; it is not the GPU-ready
tile pipeline implemented by the ParallelKittens data path.  Every record
therefore carries `"emulated_nodes": true` and must not be reported as an RDMA
or tile-overlap result:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  -m benchmarks.pdl_gemm_ar.multinode.benchmark \
  --logical-local-world-size 4 --correctness-only

OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  -m benchmarks.pdl_gemm_ar.multinode.benchmark \
  --logical-local-world-size 4 --chunk-kib 64,128,256,512,1024 \
  --depths 1,2,4,8 --warmups 20 --iterations 100 \
  --output benchmarks/pdl_gemm_ar/multinode/results/logical_2x4.jsonl
```

This experiment keeps the pinned ParallelKittens BF16 `128x256x64`, four-stage
GEMM and NVLS AllReduce arithmetic while comparing execution organization:

- upstream fused LCSC MegaKernel;
- split kernels with ordinary same-stream completion ordering;
- PDL with a whole-grid dependency wait;
- PDL admission with the original system-scope per-tile counters;
- a diagnostic independently-shaped GEMM/communication two-stream path.

See [RESULTS.md](RESULTS.md) for the first 8xH100 measurements and the current
two-stream scheduler caveat.  It also records the follow-up wave-quantization
sweep and device-side timing evidence.  The key distinction is that whole-grid
PDL only advances admission, whereas PDL plus the system-scope per-tile counters
can turn a GEMM tail into useful communication overlap.

The communication kernel is instantiated at 256, 384, 512, 768, and 1024
threads per CTA.  Optional dynamic shared-memory padding makes one communication
CTA consume more than half of an H100 SM's shared-memory capacity, providing an
admission-only way to approximate one communication CTA per SM without changing
its instructions.

The two-stream mode is not included in normal presets; request it explicitly
with `--two-stream-ctas`.  The launch records a disabled-timing event on the
main stream and makes the non-blocking communication stream wait for it.  This
rendezvous is required for honest CUDA-event timing: without it, the scheduler
may begin the communication grid before the caller's start event has executed.
A tile-counter communication CTA spins while holding its SM, so either
submission order can deadlock if the scheduler admits only one grid.  When
compute CTAs plus padded communication CTAs total 132 and both kernels are
one-CTA/SM, their observed resident role counts form a disjoint partition; this
is a role-count control, not affinity to particular physical SM IDs.

This stream-only mechanism cannot guarantee deadlock-free role reservation:
CUDA may admit all spinning communication CTAs or all producer CTAs before the
other grid.  Therefore it remains a scheduler diagnostic.  A production-quality
comparison needs CUDA green contexts/execution affinity or a persistent
cooperative admission protocol.

The ordinary split path relies on normal same-stream grid completion and then a
system-scope, eight-rank grid-ready counter before its first NVLS load.  The PDL
whole-grid path enables early admission, executes `griddepcontrol.wait` for its
same-device predecessor, and waits on the same eight-rank counter.  PDL itself
is not treated as a general inter-GPU memory primitive.  The separate PDL-tile
and two-stream modes use the original system-scope per-tile counters for
cross-GPU readiness.

The grid-ready producer is device-side: each compute CTA joins a GPU-scope
acquire/release completion chain after its release-system tile stores, and the
last CTA publishes one system-scope contribution to all eight ranks.  This is
required for correctness when a process starts with PDL; same-device
`griddepcontrol.wait` alone does not imply that the other seven GPUs finished.

## Build

Use the CUDA/PyTorch environment required by ThunderKittens:

```bash
cd benchmarks/pdl_gemm_ar
make ARCH=SM90
```

The compiler output includes ptxas register and spill reports.

## Smoke and correctness

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  benchmark.py --preset smoke --correctness \
  --m 8192 --k 1024 --n 8192 --warmups 5 --iterations 20
```

This validates fused, ordinary split, PDL whole-grid, and PDL tile-counter.
Run the diagnostic two-stream path separately and only with a timeout.

For the register-resident FP4 epilogue paths, run PRE and POST separately:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset fp4 --fp4-mode pre --correctness-only \
  --m 8192 --k 1024 --n 8192 --split-comm-ctas 8 --split-comm-threads 1024

OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset fp4 --fp4-mode post --correctness-only \
  --m 8192 --k 1024 --n 8192 --split-comm-ctas 8 --split-comm-threads 1024
```

The PRE producer-only phase-wrap regression can be minimized to one compute
CTA processing exactly three tiles on each GPU:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset fp4 --fp4-mode pre --fp4-producer-only-control \
  --fp4-producer-control-comp-ctas 1 \
  --m 128 --k 1024 --n 768 --warmups 0 --iterations 0
```

This is also the preferred Compute Sanitizer target; it preserves the first
two-phase mbarrier wrap without instrumenting all 1056 compute CTAs.

## Communication saturation sweep

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 \
  benchmark.py --preset mechanism --comm-sweep --randomize \
  --m 32768 --k 4096 --n 32768 \
  --warmups 20 --iterations 100 --output results/run.jsonl
```

For replicated scheduling comparisons, use a different `--random-seed` in each
fresh process so clock/order drift is not confounded with one mode.

## Wave-quantization sweep

For the accepted `128x256` GEMM tile on 132 H100 SMs, choose `M` values by the
remainder of `(M/128)*(N/256)` modulo 132.  With `N=8192`, the recorded points
are `M=7936,8192,8448,8960`, producing remainders `4,68,0,128`.

Run each shape in fresh processes with rotated mode orders.  For example:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset mechanism \
  --mode-order pdl_grid,pdl_tile,default \
  --split-comm-ctas 20 --split-comm-threads 1024 \
  --m 7936 --k 1024 --n 8192 \
  --warmups 20 --iterations 100 --record-samples --correctness \
  --output results/wave_rb62_order1.jsonl
```

Use a timeout for PDL-first progress testing and do not treat a single process
as a formal result.  The existing formal attempt once left rank 0 idle and
ranks 1-7 in device-side waits, although a later protected recheck completed
correctly.  PTX specifies that repeated `griddepcontrol.launch_dependents`
within one CTA has no side effect after the first, so the early-plus-fallback
arrival pattern is redundant but is not the established cause of that stall.

To verify the overlap mechanism rather than benchmark latency, repeat a short
run with `--instrument`.  Compare communication `trace_ns[*][0]` (CTA entry)
and `trace_ns[*][2]` (first task) with the maximum compute `trace_ns[*][3]`
(producer CTA exit).  Instrumentation must remain disabled for latency runs.

The sweep marks the configurations reaching at least 90% and 95% of the best
measured logical NVLS payload bandwidth.  The minimum 90% configuration is the
practical low-SM operating point; the minimum 95% configuration is the
higher-bandwidth operating point.  Use the independent GEMM-interference sweep
below as the final communication CTA calibration; the dependent two-stream
mode cannot provide a deadlock-free confirmation without hard SM partitioning.

For the required communication-under-GEMM calibration, the harness launches an
independent full-grid GEMM and an independent AllReduce on separate streams.
They use separate output tensors, so the communication grid has no data wait;
the experiment isolates multimem throughput under GEMM memory traffic.  The
communication duration is the median of per-iteration, cross-rank-max
`%globaltimer` spans from CTA entry to grid exit.  The outer CUDA-event time
reports the combined critical path:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset mechanism --sweep-only --interference-sweep \
  --interference-ctas 4,8,12,16,20,24,28,32 \
  --interference-threads 1024 --m 8192 --k 1024 --n 8192 \
  --warmups 20 --iterations 100 --record-samples \
  --output results/interference.jsonl
```

Because compute and communication each use one CTA/SM, the harness also reports
their distinct/overlapping/union SM role counts.  The launch-count cap prevents
more than 132 simultaneous roles; it does not bind roles to named physical SMs.

## Replicated path comparison

Report both the interference sweep's smallest 90%-of-peak and 95%-of-peak CTA
counts.  Use the former for the low-SM comparison and the latter for the
high-bandwidth comparison.  A Latin-square-style sequence of `--mode-order`
values avoids permanently placing one implementation first or last.  Add
`--record-samples` to retain all 100 per-iteration, cross-rank-max samples in
JSONL.

For the reported allocation (`32` fused communication roles and `20x1024`
split communication CTAs), one replicated invocation is:

```bash
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  benchmark.py --preset mechanism \
  --mode-order fused,default,pdl_grid,pdl_tile \
  --fused-comm-ctas 32 --split-comm-ctas 20 \
  --split-comm-threads 1024 --m 8192 --k 1024 --n 8192 \
  --warmups 20 --iterations 100 --record-samples \
  --output results/compare_order1.jsonl
```

Run the other fresh processes with rotated `--mode-order` values.

The persistent fused path and decoupled split paths intentionally accept
separate allocation controls: `--fused-comm-ctas` is the number of physical
communication roles in the fused kernel, while `--split-comm-ctas` and
`--split-comm-threads` configure the independently-shaped communication grid.

## Matched QuACK TensorSSA integration

`quack_pk_compare.py` uses QuACK's SM90 TensorSSA BF16-to-E2M1 epilogue as an
external producer for the same ParallelKittens PRE-FP4 consumer.  Put both
repository roots on `PYTHONPATH` and run from this directory:

```bash
PYTHONPATH=/path/to/quack:/path/to/mixture-of-kittens \
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=8 -- \
  quack_pk_compare.py --m 8192 --k 1024 --n 8192 \
  --comm-ctas 20 --comm-threads 1024 \
  --warmups 20 --iterations 100 --preheat-ms 1000 \
  --mode-order \
tk_persistent,pk_bf16_default,pk_bf16_pdl_grid,pk_bf16_pdl_tile,quack_default,quack_pdl_tail,quack_pdl_early \
  --output results/quack_pk_order1.json
```

Use `--correctness-only` first.  This checks the bit-exact data/scale planes,
all 2048 per-rank tile counters, decoded eight-rank AllReduce, and the QuACK
two-stream diagnostic.  Use `--pk-equivalence-only` to verify that persistent,
ordinary split, whole-grid PDL, and tile-PDL BF16 outputs are bit-identical to
one another; they are not required to be bit-identical to a PyTorch/NCCL SUM
whose reduction order differs.

The optional modes `quack_plain`, `pk_bf16_producer`, and `pk_fp4_producer`
provide a matched producer-only breakdown.  `quack_two_stream` and
`pk_bf16_two_stream` are progress diagnostics only and must be protected by an
external timeout.

Each standalone AllReduce sample first restores the same BF16 source tensor on
the current stream.  The restore is recorded before the timing start event, so
it is ordered but excluded from reported communication latency; this prevents
repeated in-place reductions from multiplying the data by eight every sample.
The custom NVLS collective itself couples the eight ranks.  The standalone and
interference sweeps use an NCCL barrier before each timed sample to align the
ranks, then reduce the completed timing vectors with NCCL `MAX`.

The fused/default/PDL path comparison uses a different timing helper: it does
not insert a collective between iterations.  It reduces the completed sample
vector with NCCL `MAX` only after every custom NVLS operation has finished,
avoiding an NCCL/custom-counter interleaving in the critical comparison.

## Profiling

Latency runs leave device instrumentation disabled.  For `%globaltimer` and
`%smid` records, run a short, separate invocation with `--instrument`:

```bash
nsys profile --trace=cuda,nvtx -o pdl_gemm_ar \
  torchrun --standalone --nproc-per-node=8 -- benchmark.py \
  --preset smoke --m 8192 --k 1024 --n 8192 \
  --warmups 2 --iterations 3 --instrument
```

Use `resource_report()` output, ptxas logs, and Nsight Compute together.  Hopper
warp-specialized `setmaxnreg` targets are not fully represented by a single
`cudaFuncAttributes.numRegs` value.

## Interpretation constraints

- PDL controls successor admission only; cross-GPU tile visibility still uses
  ParallelKittens' `red.release.sys` signal and system-scope polling.
- Whole-grid PDL and tile-counter PDL are separate modes.
- Communication CTA count, distinct observed SM count, and fused persistent SM
  roles are reported separately.
- `fused` is the upstream ParallelKittens/ThunderKittens LCSC implementation;
  split paths reuse the same ThunderKittens arithmetic.  QuACK TensorSSA is a
  separate DSL baseline and must be labeled separately when its wire or timing
  scope differs.
- Every latency sample is reduced with `MAX` across the eight ranks before the
  median/mean/p95 are computed.
- Shape alternatives in `resource_model.py` are analytical diagnostics, not
  performance-equivalent replacements for the accepted four-stage kernel.

## Two-node hierarchical BF16 experiment

The multi-node code is under `multinode/`; its protocol and measurement
contract are documented in `multinode/DESIGN.md`.  Compression is deliberately
excluded from this first gate.  The path under test is:

```text
PK GEMM tile -> NVLS owner pack -> NCCL/UCCL owner lane -> NVLS multicast
```

The launcher is preflight-only unless both an explicit run flag and a reviewed
approval file are present.  Run it from either node with the same repository
SHA and explicit NIC variables:

Do not invoke these commands until execution has been explicitly approved.
The first approved steps are local syntax/unit checks and a PTXAS build gate;
two-node preflight comes only after those pass.

```bash
NODE_RANK=0 MASTER_ADDR=<node0> \
NCCL_SOCKET_IFNAME=<ifname> NCCL_IB_HCA=<hca-list> NCCL_IB_GID_INDEX=<gid> \
bash benchmarks/pdl_gemm_ar/multinode/run_2node.sh
```

After collecting both node snapshots, compare them on a review machine:

```bash
python -m benchmarks.pdl_gemm_ar.multinode.compare_preflight \
  /path/preflight_node0.json /path/preflight_node1.json \
  --output /path/preflight_approval.json
```

Only after approval, pass the same approval JSON to both nodes.  The first run
is correctness only, followed by a separate stream-order instrumentation gate,
latency, and isolated stages.  Use `TRIAL_ID=0,1,2` in distinct latency
torchrun processes; the launcher derives a different randomized mode order for
each trial and only global rank 0 writes a JSONL file:

```bash
RUN_BENCHMARK=1 RUN_PHASE=correctness \
PREFLIGHT_APPROVAL_FILE=/path/preflight_approval.json \
bash benchmarks/pdl_gemm_ar/multinode/run_2node.sh

RUN_BENCHMARK=1 RUN_PHASE=latency TRIAL_ID=0 \
PREFLIGHT_APPROVAL_FILE=/path/preflight_approval.json \
bash benchmarks/pdl_gemm_ar/multinode/run_2node.sh

RUN_BENCHMARK=1 RUN_PHASE=stages WINDOW_TILES=4 MAX_INFLIGHT=1 \
PREFLIGHT_APPROVAL_FILE=/path/preflight_approval.json \
bash benchmarks/pdl_gemm_ar/multinode/run_2node.sh

RUN_BENCHMARK=1 RUN_PHASE=instrument MODES=pdl_tile \
WINDOW_TILES=4 MAX_INFLIGHT=1 WARMUPS=2 ITERATIONS=3 \
PREFLIGHT_APPROVAL_FILE=/path/preflight_approval.json \
bash benchmarks/pdl_gemm_ar/multinode/run_2node.sh
```

`run_2node_uccl_plugin.sh` uses the identical matrix through the UCCL NCCL net
plugin.  Set `UCCL_PLUGIN_PATH` to the absolute plugin binary before preflight;
the launcher assigns the same path to `NCCL_NET_PLUGIN`, and the approval binds
its SHA256.  The stage decomposition records local GEMM, NVLS owner pack,
GPU-ready control, owner-lane network, NVLS unpack, and the node-local completion
join.  It also measures empty window orchestration so that overhead is not
triple-counted across ready/network/unpack stages.  Its ideal-overlap value is
an analytical estimate, not a formal bound or measured latency.
