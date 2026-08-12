# Hierarchical BF16 experiment result

## Claim

State one measured claim.  Negative results are first-class claims.

## Scope

- Git SHA:
- ThunderKittens submodule SHA:
- topology:
- GPU/NIC:
- backend and plugin SHA:
- dtype/shape:
- clock/power state:
- timing scope:
- preflight comparison JSON:
- approval generated before this run: yes/no
- explicit user execution approval obtained: yes/no
- trial IDs and fresh-process count:
- NCCL `Work.wait()` stream-order contract validated: yes/no
- NVLS multicast GPU-join completion validated: yes/no
- all `unpack_after_remote_by_window` values true: yes/no
- `join_after_all_unpacks` true: yes/no
- Nsight shows every unpack after its actual NCCL kernel completes: yes/no

## Commands

```bash
# preflight

# correctness

# latency

# instrumentation, if separate

# isolated stage decomposition
```

## Results

Link the committed JSON/JSONL summary and raw samples.  Store large Nsys/NCU
reports as release artifacts and record their SHA256 here.

## Complexity delta

- changed LOC:
- synchronization states:
- CUDA streams:
- NCCL lane communicators:
- host progress threads:
- required environment variables:

## Stage decomposition

- local GEMM:
- empty window orchestration:
- NVLS owner pack:
- GPU ready control:
- inter-node owner-lane network:
- NVLS unpack:
- node-local completion join:
- isolated-stage sum:
- estimated ideal-overlap latency (not a formal bound):
- measured full PDL path:
- flat NCCL path:
- perfect-overlap opportunity versus flat:

The ideal-overlap estimate is a model, not a measured result.  Record whether the owner
tile production order actually matches the assumed uniform window schedule.

## Conclusion

Record what the data supports, the confidence level, and what it does not
support.  Logical-node experiments must not be presented as RDMA results.
