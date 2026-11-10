---
name: discover-training-system
description: Establish what the training system really is before optimizing it - GPU and interconnect topology, NUMA and pinned-memory limits, effective environment, and which code path actually executes.
license: Apache-2.0
when_to_use: Start of every task; a receipt contradicts the expected hardware behaviour; a flag or env change shows no effect; ranks on some nodes are consistently slower; 'is IB used', 'NVLink topology', 'NUMA', 'pinned memory', 'why did the flag do nothing'.
metadata:
  author: InfraRSI A0 (human-authored)
  source: human-authored (A0)
  version: 1
  tags: [discover, hardware, topology, nvlink, infiniband, numa, single-node, multi-node, h100-80gb]
---

# Discover the training system

The HardwareProfile from `discover` is read-only input (read_env). This skill says how to interpret it and which
facts must be confirmed from receipts rather than assumed.

## Trigger conditions
- First turn of a task (read_env before the first proposal).
- A configuration change produced no StepTime change at all (suspect the setting never reached the executed path).
- Persistent stragglers: `per_step[].slowest_rank` keeps naming ranks on one node or one socket.
- Multi-node tasks, any EP/DP traffic crossing nodes.

## Required observations (receipt fields)
- `env` (versions, hostnames, GPU names, whitelisted env as actually applied), `nodes`, `world_size`.
- `per_step[].slowest_rank`, `memory.per_rank[]` (rank spread), `phases.startup/load/build`.
- HardwareProfile: GPU model and memory, `nvidia-smi topo -m` matrix, IB port state/rate, NUMA nodes per socket, RAM, memlock, software versions.

## Rules: topology
- WINH nodes: 8x H100 80GB, every GPU pair NV18 (18 NVLink links per GPU at 26.562 GB/s each); eight 400 Gb/s IB HCAs Active/LinkUp. Keep TP and the largest all-to-all (EP) inside a node first; cross-node traffic is DP/EDP gradient reduction or EP across IB.
- Measured NCCL bus bandwidth at 256 MiB (inter-node runs, NVLink included, not NIC line rate): all-reduce 234.47 GB/s, all-gather 139.11, reduce-scatter 135.19, all-to-all 44.96 GB/s. All-to-all is the expensive collective; size EP accordingly.
- `ibdev2netdev` may print nothing and NVLink traffic counters may be unsupported by the installed nvidia-smi; absence of a counter is not absence of the link. Confirm IB use from NCCL init lines (NET/IB with GPUDirect RDMA), not from tool output gaps.
- The shell can see fewer CPUs than `lscpu` reports (affinity/cgroups: 168 vs 192 on several WINH nodes). Size OMP_NUM_THREADS from the affinity mask, not from lscpu.

## Rules: NUMA and pinned memory
- Some nodes run sub-NUMA clustering (2 NUMA nodes per socket), others one. Pinned host buffers allocated without binding can land on the remote NUMA node and slow host-side optimizer work or H2D/D2H copies on a subset of ranks.
- SSH `ulimit -l` does not prove the training process can pin; a RAM snapshot is not a reservation. Large core dumps in /dev/shm once consumed about 0.5 TB per node and pushed pinned memory remote; core files must be disabled.

## Rules: installed is not executed
- The effective environment can differ from the job literal: a launch shell set CUDA_DEVICE_MAX_CONNECTIONS=1 over a job value of 8. Read the applied env from the receipt.
- A feature flag without a consumer does nothing ("pseudo" variants passed review because their flags were never read). Prove execution by call counts or a changed kernel trace, not by the flag value.
- A dispatch mode or wrapper can silently route a fused kernel to a dense eager fallback. If a fused path shows exactly the old time, assume a fallback until a count proves otherwise.

## Allowed actions
- read_env; read_receipt and profile_ranges on the start candidate before proposing.
- L0 env: OMP_NUM_THREADS (host threads, relevant with optimizer offload), NCCL_* performance items of the whitelist. Topology selection (NCCL_IB_HCA, NCCL_SOCKET_IFNAME) is owned by discover, not by the agent.
- write_memory: record topology facts that explained a result (with hardware tags).

## Failure signals
- A change with an expected mechanism leaves StepTime and every phase unchanged (flag not consumed).
- `slowest_rank` concentrated on ranks of one node or socket; reserved memory identical but optimizer phase slower on some ranks.
- NCCL init falling back to sockets on a multi-node task.

## Verification
- Same-rank, same-step comparison of phases before and after (profile_ranges), plus the whole-update StepTime beyond the margin.
- For an execution-path claim: a count (calls of the new kernel per update) or a receipt field that only the new path writes.

## Success case (real)
- Flagship (32x H100, real SFT). WINH4/5 run two NUMA nodes per socket, WINH6/7 one; unbound pinned expert state landed on the remote NUMA node and made some ranks' optimizer 1-1.5 s slower. Binding each rank to its GPU's socket (sched_setaffinity) with MPOL_INTERLEAVE before CUDA init: 14.485 -> 14.059 s/step, optimizer phase 3.388 -> 3.082 s, all ranks' optimizer within 2.96-3.06 s, bitwise identical scalars. (reports/flash_next_mfu_20261003/PROGRESS.md, batch 3)

## Failure case (real)
- NeMo CP4 run had `qsa_routes_fast=true`, yet shards carrying a query-position offset or a global length fell back to the original selector (336 nonzero, 179 scalar reads on one 8,192-query shard). The flag alone produced none of the later gain; only fixing the fallback condition did. Lesson: verify the executed path per shard/rank, not the switch. (reports/agentic_moe_sft_paper_20261018/MANUSCRIPT.md 4.1, 6.2)
