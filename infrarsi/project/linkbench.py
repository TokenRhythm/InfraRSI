"""The job's own all-reduce bandwidth (textbook designs P1-3b2 / P2-2): an NCCL all-reduce of 256 MiB over every GPU of
a multi-node job, launched on its hosts with its environment after its reference run (runner.bench_links). Rank 0
prints one line `__linkbench__ {json}` with nccl-tests' bus bandwidth: 2 (n - 1) / n x size / time, in GB/s, the
slowest rank's time. Run as a script under torch.distributed.run; `parse` reads its line from the log."""
import json
import os
import re
import sys
import time

MARK, BUSY = '__linkbench__', '__linkbench_busy__'
LINE = re.compile(re.escape(MARK) + r' (\{.*\})')


def parse(lines) -> dict | None:
    """The benchmark's result from its output lines, else None."""
    for text in lines:
        m = LINE.search(text)
        if m:
            try:
                out = json.loads(m.group(1))
            except ValueError:
                return None
            return out if isinstance(out.get('busbw_gbs'), (int, float)) and out['busbw_gbs'] > 0 else None
    return None


def busy(local_rank: int) -> list[str]:
    """Other processes on the GPU this rank was given (its own environment's mask: the login shell's, the job's):
    nvidia-smi's compute apps on that GPU's uuid, this process excluded."""
    import subprocess
    import torch
    uuid = str(torch.cuda.get_device_properties(local_rank).uuid).lower().replace('gpu-', '')
    out = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], capture_output=True,
                         text=True, timeout=60).stdout
    return [ln.strip() for ln in out.splitlines() if uuid and uuid in ln.lower()
            and ln.split(',')[-1].strip() != str(os.getpid())]


def main(size_mib: int = 256, warmup: int = 5, iters: int = 20) -> int:
    import torch
    import torch.distributed as dist
    lr = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(lr)
    others = busy(lr)
    if others:                                       # another job's GPU: leave before NCCL; torchrun stops the rest
        print(f'{BUSY} rank {os.environ.get("RANK")}: {others[0]}', flush=True)
        return 3
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    x = torch.ones(size_mib * 2 ** 20 // 2, dtype=torch.bfloat16, device='cuda')
    for _ in range(warmup):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x)
    torch.cuda.synchronize()
    dt = torch.tensor([time.perf_counter() - t0], device='cuda', dtype=torch.float64)
    dist.all_reduce(dt, op=dist.ReduceOp.MAX)                # the slowest rank's time
    per = dt.item() / iters
    size = x.numel() * x.element_size()
    if rank == 0:
        print(f'{MARK} ' + json.dumps({'busbw_gbs': round(size * 2 * (world - 1) / world / per / 1e9, 2),
                                       'algbw_gbs': round(size / per / 1e9, 2), 'world': world, 'bytes': size,
                                       'nnodes': int(os.environ.get('GROUP_WORLD_SIZE', 0)) or None,
                                       'nccl': {k: v for k, v in os.environ.items() if k.startswith('NCCL_')}}),
              flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    sys.exit(main())
