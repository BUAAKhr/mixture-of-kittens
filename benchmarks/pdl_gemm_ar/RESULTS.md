# 8xH100 results and current conclusions

Environment used on 2026-08-09/10:

- 8x NVIDIA H100 SXM 80 GB, NV18 fabric;
- PyTorch 2.10.0+cu128;
- CUDA toolkit / nvcc 12.8;
- ThunderKittens `1c3920d993404dd49a6d4c7267ea11d583bd5c68`;
- BF16 `M=8192, K=1024, N=8192`.

An early separate five-sample correctness run compared fused, ordinary split,
PDL whole-grid, and PDL tile-counter against a deterministic PyTorch +
distributed AllReduce oracle with zero observed absolute difference.  The
final audit uses the stronger path-equivalence and wire checks documented in
the QuACK integration section: all native BF16 paths are bit-identical to one
another, while their shared reduction order can differ from PyTorch/NCCL BF16
SUM.  The two-stream path remains diagnostic-only; see below.

## Correctness bugs found during implementation

Two PDL details materially changed the interpretation of the early results.

First, `griddepcontrol.launch_dependents` is a per-CTA arrival.  The first
version called it only from the producer loader lane.  Every compute CTA now
executes a fallback arrival at CTA completion; without that fix, a process whose
first mode was PDL could produce large wrong-output regions.

Second, `griddepcontrol.wait` covers only the same-device predecessor.  It does
not prove that the other seven GPUs have completed their GEMMs before an NVLS
load.  Ordinary split and PDL whole-grid now use a system-scope eight-rank
grid-ready gate.  Each compute CTA joins a GPU-scope acquire/release completion
chain after its release-system output publication; the last CTA publishes one
system-scope rank contribution.  PDL tile-counter instead retains the original
per-tile release/acquire protocol.

The earlier sub-millisecond/low-millisecond numbers collected before these
fixes are retained as debugging artifacts only and are not headline results.

## Fused FP4 epilogue status

The PRE path now quantizes the final WGMMA FP32 accumulator directly from
registers.  For every logical 16-element output block it first rounds values to
BF16, reduces `amax` across the four lanes that jointly own the block, computes
a BF16 `amax / 6` decode scale, encodes E2M1 nibbles with round-to-nearest
division, and writes packed data plus scales to the current rank's local wire
allocation.  No BF16 output staging/reload is used for the quantization step.
The compiled PRE compute kernel remains at 168 registers/thread with zero
stack and zero spills.

The owner layout is deliberately sharded, not eightfold replicated:

- PRE stores each rank's local GEMM tile in that rank's wire allocation and
  the communication kernel reads all eight peer allocations;
- POST performs BF16 NVLS reduction first, assigns each output tile to one
  communication rank (`task_id % 8`), and multicasts only that owner's packed
  tile so every GPU observes the same owner-sharded wire.

Two correctness issues were found while making these paths repeatable.  First,
the PRE direct-output path removed the SMEM store but left a two-phase
`outputs_arrived` mbarrier with no backpressure.  Once a CTA processed three or
more tiles, the two consumer warpgroups could flip parity twice before the
storer observed the previous phase, leaving the storer waiting forever.  The
fix adds an `outputs_finished` acknowledgement while preserving one tile of
lookahead: consumers may compute and quantize tile `t+1` while the storer
publishes tile `t`, but they cannot signal `t+1` until `t` was observed.
Second, POST paired scales with a full-mask warp shuffle executed only by the
two writer lanes, which is undefined.  All lanes now execute an XOR shuffle
within each 16-lane group before lanes 0 and 16 perform the 32-bit multicast
store.

The phase-wrap boundary is directly exercised by `M=1152,K=1024,N=8192`, where
some of the 132 compute CTAs process three tiles.  It now completes with every
tile counter exactly 8.  The former failing `M=4096` and `M=8192` producer-only
cases also complete, with all counters exactly 8.  At full
`M=8192,K=1024,N=8192`, default, whole-grid PDL, and tile-PDL report all of:

```text
fp4_data_mismatch       = 0
fp4_scale_max_abs_diff  = 0
max_abs_diff            = 0
mean_abs_diff           = 0
```

This holds for both PRE and POST.  PRE additionally passed two warmups plus
five timed replays in each mode.  A host-barrier decomposition (producer only,
host/NCCL barrier, communication only) also reports a bit-exact PRE wire and
zero final error.

Compute Sanitizer 2025.2 results for the first three-tile phase-wrap case are:

- `synccheck`: 0 errors;
- `memcheck`: 0 errors;
- `initcheck`: 0 errors;
- `racecheck`: 0 errors and 0 warnings on a minimized one-CTA-per-GPU case in
  which that CTA processes exactly three tiles.

The full 132-CTA-per-GPU racecheck did not finish inside five minutes and is
not claimed as a pass.  The minimized case preserves the specific parity-wrap
failure while reducing instrumented CTAs from 1056 to 8.

No FP4 latency conclusion is reported yet.  GPU 4 had an unrelated process and
the node was not clock-controlled during these correctness runs.  The observed
roughly 2.6--2.7 ms samples are stability evidence only, not a comparison of
PRE, POST, PDL, or fused baselines.

## Compiled resources and GEMM tile model

- split compute: 384 threads, 168 registers/thread, 231424 B requested dynamic
  SMEM, zero spills, one CTA/SM;
- fused LCSC control: 168 registers/thread, 231424 B dynamic SMEM, 4-byte spill
  store and 4-byte spill load;
- communication, non-instrumented: 38 regs at 256 threads, 40 regs at 384,
  38 regs at 512, 32 regs at 768, and 28 regs at 1024; zero spills;
- 128 KiB communication SMEM padding permits at most one communication CTA/SM.

The accepted GEMM tile is `128x256x64`, four stages.  Its analytical live SMEM
is 208 KiB (`3 * 48 KiB + 64 KiB`); the launch requests nearly all available
SMEM because the upstream fused scheduler reuses the remaining allocation.
The accumulator alone is 128 FP32
registers per consumer thread / 32768 registers per CTA.  The pinned
warp-specialized targets are 40 registers for the producer warpgroup and 232
for each of two consumer warpgroups: 64512 of H100's 65536 registers/SM, leaving
only 1024 registers.  Two-CTA residency would require the consumer target to be
at most 108 registers/thread analytically, or 104 after the SM90 eight-register
per-thread allocation quantum.  The accepted tile therefore cannot co-reside
with another accepted compute CTA.

## Communication saturation

The no-interference standalone sweep found 16 x 1024-thread CTAs sufficient for
the best measured range.  The required communication-under-GEMM calibration is
slightly stricter.  It launches independent GEMM and AllReduce grids on two
streams, uses separate outputs, and records a `%globaltimer` communication-grid
span for every iteration.  The table uses the median of each 100-sample span
vector after sample-wise NCCL `MAX` across the eight ranks.  Each configuration
used 20 warmups and 100 timed iterations.

| Comm CTAs | Compute roles | Comm grid | Logical BW | Peak fraction | Role overlap |
|---:|---:|---:|---:|---:|---:|
| 4 | 128 | 1.316 ms | 102.0 GB/s | 37.33% | 0 |
| 8 | 124 | 0.733 ms | 183.0 GB/s | 66.95% | 0 |
| 12 | 120 | 0.574 ms | 234.0 GB/s | 85.61% | 0 |
| 16 | 116 | 0.518 ms | 258.9 GB/s | 94.72% | 0 |
| 20 | 112 | 0.500 ms | 268.6 GB/s | 98.29% | 0 |
| 24 | 108 | 0.493 ms | 272.1 GB/s | 99.58% | 0 |
| 28 | 104 | 0.500 ms | 268.6 GB/s | 98.29% | 0 |
| 32 | 100 | 0.491 ms | 273.3 GB/s | 100.00% | 0 |

Thus `20 CTA x 1024 threads` is the smallest tested configuration meeting the
95%-of-peak criterion under GEMM interference.  This is the quantitative split
communication allocation used below.  The observed role sets were disjoint and
covered all 132 SMs for every sweep point.

## Historical fused versus split/PDL scheduling

The comparison uses 32 persistent communication roles for the upstream fused
LCSC control, matching the cited baseline's best allocation, and 20 x
1024-thread communication CTAs for the independently-shaped split paths.  Each
fresh process used 20 warmups and 100 samples; the order was rotated.  Three
completed replications are summarized by the median of their per-process
medians:

| Mode | Median latency | Delta vs fused | Three-process range |
|---|---:|---:|---:|
| fused LCSC, 32 comm roles | 0.841 ms | reference | 0.618-0.841 ms |
| PDL tile-counter, 20x1024 | 0.884 ms | +5.11% | 0.695-0.888 ms |
| PDL whole-grid, 20x1024 | 0.965 ms | +14.80% | 0.724-0.968 ms |
| ordinary split, 20x1024 | 0.970 ms | +15.40% | 0.727-0.971 ms |

This table predates the final matched cross-DSL campaign below.  It remains
useful mechanism evidence, but its absolute latencies are not the final
headline numbers because the operating point was unstable.  For the accepted
comparison, use the later `0.608 ms` TK persistent and `0.699 ms` PK BF16
tile-PDL results at the refreshed 20-CTA point.

The host did not lock GPU clocks, and the raw samples contain obvious operating
point changes (roughly 0.62/0.84 ms for fused and 0.72/0.97 ms for whole-grid
split), plus GPU 4 had an unrelated low-utilization process using about 11 GiB
at the final audit.  Therefore the absolute range is wide.  The stable paired
conclusions are:

1. whole-grid PDL is essentially neutral for this workload: its median advantage
   over ordinary split is about 0.5%;
2. per-tile readiness is the useful mechanism here, improving the split path by
   about 9% and closing most of the gap to fused;
3. even with independently optimal communication CTA shape/count, the fused
   persistent kernel remains about 5% faster than PDL tile-counter at this
   shape.

This differs from the decode PDL reconstruction for a concrete reason: the
AllReduce consumer has no long independent weight/KV prologue to execute before
its activation wait.  Whole-grid PDL can move admission but has little useful
work to hide.  Tile counters narrow the real cross-GPU data dependency and do
recover meaningful overlap.

## Two-stream limitation

The stream-only dependent two-kernel path cannot implement a hard SM partition.
CUDA may admit all 132 one-CTA/SM compute blocks before any spinning
communication block, or admit the spinning communication grid without the
producer set needed to make progress.  Both submission orders were observed to
deadlock in valid-looking configurations.  Launch order and dynamic-SMEM
padding are therefore not substitutes for reservation.

The two-stream code remains available only with explicit `--two-stream-ctas`
for scheduler diagnostics.  A fair, deadlock-free headline comparison requires
CUDA green contexts/execution affinity or a cooperative persistent admission
protocol.  The independent interference sweep is valid because its AllReduce
has no producer dependency and therefore cannot spin waiting for GEMM.

## Practical conclusion

For this ParallelKittens GEMM + NVLS AllReduce case, decoupling resource shapes
is valuable: a 1024-thread communication CTA needs only 28 registers and
20 CTAs are enough to reach 98.3% of measured communication peak under GEMM
traffic.  The accepted 128x256 GEMM tile, however, already consumes almost the
entire H100 register file and 208 KiB of pipeline SMEM, so software cannot make
two accepted GEMM CTAs co-resident without changing its throughput-critical
tile/pipeline.

PDL's available space is correspondingly narrow.  Whole-grid PDL does not beat
ordinary completion in a meaningful way; PDL plus genuine tile-level
cross-GPU readiness gets within about 5% of the fused persistent path.  The
remaining gap is consistent with persistent role scheduling and state/resource
lifetime, not CPU launch overhead.

## Wave quantization follow-up

The main `M=8192` comparison is already a wave-quantized GEMM.  With the
accepted `128x256` output tile and 132 one-CTA/SM compute CTAs,

```text
total_tiles = (M / 128) * (N / 256)
remainder   = total_tiles mod 132
early_free  = remainder == 0 ? 0 : 132 - remainder
```

For `M=N=8192`, `total_tiles=2048=15*132+68`.  Sixty-four compute
CTAs therefore execute 15 tiles while 68 execute 16; up to 64 SMs can become
available roughly one tile before the producer grid finishes.  PDL does not
remove this GEMM tail.  It can admit a successor into the holes created by the
tail.

Whether early admission is useful depends on the successor's dependency
granularity:

- whole-grid PDL admits communication CTAs early, but they immediately execute
  `griddepcontrol.wait` and then the eight-rank grid-ready gate.  The AllReduce
  has no independent weight/KV-load prologue, so admission alone exposes almost
  no useful work;
- tile-counter PDL admits the same CTAs early, but each CTA waits only for the
  first output tile assigned to it.  A ready tile can execute multimem reduction
  while the producer's last CTAs are still computing other tiles.

### Directional remainder sweep

We held `N=8192`, `K=1024`, and the split communication launch at
`20x1024`, and changed `M` to create four different remainders.  These runs used
two warmups, five timed samples, deterministic correctness, and a fixed mode
order.  They are mechanism evidence, not publication-quality latency results:
the host was not clock-locked and exhibited large operating-point changes.

| M | Total tiles | mod 132 | CTAs that can retire one tile early | Whole-grid PDL gain vs default | Tile-PDL gain vs default |
|---:|---:|---:|---:|---:|---:|
| 7936 | 1984 | 4 | 128 | 0.52% | 4.40% |
| 8192 | 2048 | 68 | 64 | -0.10% | 4.10% |
| 8448 | 2112 | 0 | 0 | 0.76% | 3.03% |
| 8960 | 2240 | 128 | 4 | 0.38% | 3.41% |

All four shapes reported zero maximum absolute error for default, whole-grid
PDL, and tile-PDL.  The ordering is consistent with the hypothesis: the most
severe short tail has the largest tile-PDL gain, while the exact-wave point has
the smallest.  Four shapes and five samples are insufficient to fit a robust
quantitative law, so the current claim is correlation plus mechanism, not a
completed formal sweep.

### Device-side timing evidence

Separate one-sample instrumented runs recorded `%globaltimer` at compute CTA
entry/first load/first store/exit and communication CTA entry/wait completion/
first task/exit.  Instrumentation is disabled for all latency tables.

| Shape | Mode | Comm CTAs entering before producer grid end | Comm CTAs starting first task before producer grid end | Earliest first-task lead |
|---|---|---:|---:|---:|
| M=7936, severe short tail | whole-grid PDL | 20/20 | 0/20 | none |
| M=7936, severe short tail | tile-PDL | 20/20 | 20/20 | 20.736 us |
| M=8448, exact wave | whole-grid PDL | 20/20 | 0/20 | none |
| M=8448, exact wave | tile-PDL | 20/20 | 20/20 | 9.472 us |

The exact-wave case can still have a small natural straggler tail because CTA
durations and admission are not identical.  The important comparison is that
the severe quantized tail roughly doubles the observed first-task overlap
window.  Whole-grid PDL proves early admission but not early useful work;
tile-PDL proves both.

### Progress anomaly and semantic audit

An attempted three-process formal wave sweep completed its first `M=7936`
process (`default 1.258544 ms`, `pdl_grid 1.255456 ms`, `pdl_tile 1.173776 ms`)
but the raw samples jumped among roughly 0.7, 1.25, and 3.1 ms operating points.
The next process, which started with `pdl_grid`, once left rank 0 idle while
ranks 1-7 remained in device-side waits.  That process was terminated, so the
formal sweep is incomplete and its single-process latency is not a headline
result.

A later timeout-protected `pdl_grid -> pdl_tile -> default` recheck at
`M=7936` completed on all eight ranks, with zero maximum absolute error in all
three modes.  Its five-sample medians were 0.984896, 0.899392, and 0.955520 ms,
respectively.  Because GPU 0 and GPU 4 had unrelated activity and clocks were
not controlled, this recheck establishes correctness/progress only; it does
not resolve whether the earlier stall was a transient rank failure, NVLS
progress issue, or another environmental fault.

The producer currently executes an early `griddepcontrol.launch_dependents`
after its first TMA issue and a fallback invocation at CTA completion.  PTX ISA
states that repeated invocations by a CTA have no additional side effects after
the first, so this is redundant but is not a valid explanation for the stall.
The fallback remains useful for CTAs with no assigned tile.  The important
correctness fixes remain the eight-rank system-scope grid gate and the per-tile
release/acquire counters described above.

### Current wave conclusion

Wave quantization increases the opportunity available to PDL, but only if the
dependent kernel can consume partial readiness.  For this GEMM + NVLS
AllReduce, the useful design is therefore **PDL admission plus tile-level
system-scope counters**, not whole-grid PDL.  The theoretical opportunity is
bounded by the duration of approximately one producer tile for static-stride
CTAs and by the communication work that fits in the SM holes.  It cannot exceed
the remaining split-versus-fused gap indefinitely: after the tail is hidden,
persistent role scheduling, state lifetime, counter/polling cost, and NVLS
contention remain.

The next formal experiment should run the four remainder points in rotated
fresh processes on a quiescent, clock-controlled node, then repeat the timing
instrumentation per point.  A second axis should vary `K`, because a longer
per-tile GEMM increases the tail window without changing the tile remainder;
this separates wave geometry from tile duration.

## QuACK TensorSSA E2M1 integration

The QuACK producer is now connected directly to the ParallelKittens PRE-FP4
consumer through the same VMM-backed data, scale, and tile-counter planes.
QuACK performs BF16 rounding, 16-element scale reduction, E2M1 encoding, and
packing from the final TensorSSA accumulator, uses SMEM only to regularize the
full `128x256` tile for asynchronous TMA stores, waits for destination
completion, and then publishes one system-scope release per rank-local tile.
The PK consumer acquires all eight rank counters for its assigned tile.

At `M=N=8192, K=1024`, all 2048 counters were exactly one, all wire mismatch
counts were zero, and the decoded AllReduce output had zero max/mean error for
ordinary completion, tail-trigger PDL, early-trigger PDL, and the diagnostic
two-stream path.  The minimized eight-rank `synccheck` run reported zero
errors.  Native PK PRE-FP4 default/grid-PDL/tile-PDL correctness also passed.

The communication-under-GEMM sweep was refreshed before timing.  `16x1024`
remains the minimum tested 90%-of-peak configuration at `94.78%`; `20x1024`
remains the minimum tested 95%-of-peak configuration at `98.38%`.

Each table value is the median of three fresh-process medians.  A process used
1000 ms preheat, 20 warmup rounds, 100 timed rounds interleaved at single-launch
granularity, rotated mode order, and sample-wise MAX over eight ranks.

| Path | 16 CTA | 20 CTA |
|---|---:|---:|
| TK persistent BF16 LCSC | 0.609 ms | 0.608 ms |
| PK BF16 ordinary split | 0.729 ms | 0.728 ms |
| PK BF16 whole-grid PDL | 0.728 ms | 0.728 ms |
| PK BF16 tile-PDL | 0.705 ms | 0.699 ms |
| PK register-FP4 ordinary split | 1.528 ms | 1.301 ms |
| PK register-FP4 whole-grid PDL | 1.522 ms | 1.300 ms |
| PK register-FP4 tile-PDL | 1.398 ms | 1.264 ms |
| QuACK TensorSSA ordinary split | 1.221 ms | 1.072 ms |
| QuACK TensorSSA tail-trigger PDL | 1.222 ms | 1.070 ms |
| QuACK TensorSSA early tile-PDL | 1.198 ms | 1.049 ms |
| QuACK TensorSSA producer only | 0.390 ms | 0.390 ms |

The scheduling conclusion is unambiguous.  Whole-grid PDL is neutral, and a
tail trigger is neutral.  Early admission becomes useful only when paired with
tile readiness: it improves QuACK by `1.91%`/`2.12%`, native BF16 split by
`3.33%`/`4.06%`, and native register-FP4 split by `8.48%`/`2.87%` at 16/20
CTAs respectively.

TensorSSA fusion materially improves the FP4 producer but does not yet beat
BF16 end to end.  A matched producer breakdown at the 20-CTA operating point
measured QuACK plain BF16 GEMM at `0.200 ms`, PK/TK plain BF16 GEMM at
`0.223 ms`, QuACK E2M1 producer at `0.387 ms`, and PK/TK register-E2M1 producer
at `0.701 ms`.  QuACK early tile-PDL is therefore `17.0%` faster than PK's
register-FP4 tile-PDL, but `50.1%` slower than PK BF16 tile-PDL and `72.4%`
slower than persistent BF16.  The remaining problem is codec/decode cost, not
PDL placement.

The two-stream diagnostic completed (`0.568--0.569 ms` PK BF16 and
`1.054--1.218 ms` QuACK), but remains non-headline because streams cannot
reserve a hard SM partition and valid schedules can deadlock spinning
consumers.  Use green contexts/execution affinity or cooperative persistent
admission before treating it as a production result.

Finally, persistent/default/grid-PDL/tile-PDL BF16 device outputs are
bit-identical to one another.  Their common reduction order differs from
PyTorch/NCCL BF16 SUM: max absolute difference `0.125`, mean `0.00647`.  The
old `atol=0.02` oracle failure is therefore a reference-order/tolerance issue,
not a PDL correctness discrepancy.
