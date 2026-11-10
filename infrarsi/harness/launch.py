"""torchrun command lines, environment and per-node wrapper specs (INTERFACES section 3.3)."""
from __future__ import annotations

import fnmatch
import hashlib

from infrarsi.contracts import RemotePaths

ENTRY_MODULE = 'infrarsi.adapters.megatron_core.generic_train'
VENV_PYTHON = f'{RemotePaths.venv}/bin/python'
WRAPPER_PYTHON = '/usr/bin/python3'
FN = RemotePaths.flash_next
DEEPEP_ROOT = f'{FN}/code/vendor/DeepEP-fp32-combine'
# Order from flash-next/scripts/launch.sh: compat -> runtime -> bridge src -> vendored Megatron-Core -> DeepEP -> overlay.
MCORE_STACK = [f'{FN}/code/vendor/compat', f'{FN}/code/runtime', f'{FN}/code/vendor/mcore-bridge/src',
               f'{FN}/code/vendor/Megatron-Core', DEEPEP_ROOT,
               '/mnt/base/qwen-sft/runtime/flash-training-ad-20260907/overlay']
# Site/operational settings shared by every arm (not performance knobs). They are applied last, so neither the
# request's whitelisted env nor anything else in the spec can override them; NCCL and the Gloo groups (torch.distributed
# store/barriers, the lifecycle supervisor) are both bound to eth0 (INTERFACES 6: multi-node launches set
# GLOO_SOCKET_IFNAME; single-node launches get the same binding, as T-train's launch_dev.sh does).
SITE_ENV = {'NCCL_SOCKET_IFNAME': 'eth0', 'GLOO_SOCKET_IFNAME': 'eth0', 'NCCL_NET_PLUGIN': 'none', 'USE_MCORE_GDN': '1',
            'DEEPEP_ROOT': DEEPEP_ROOT,
            'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
            'TORCHINDUCTOR_COMPILE_THREADS': '1', 'TOKENIZERS_PARALLELISM': 'false'}
# Design section 1.8: allocator, CUDA_DEVICE_MAX_CONNECTIONS, NCCL performance items, OMP threads.
ENV_WHITELIST = ['PYTORCH_CUDA_ALLOC_CONF', 'CUDA_DEVICE_MAX_CONNECTIONS', 'OMP_NUM_THREADS', 'NCCL_*']
ENV_DENY = ['NCCL_SOCKET_IFNAME', 'NCCL_NET_PLUGIN', 'NCCL_DEBUG*', 'NCCL_TOPO_FILE', 'NCCL_IB_DISABLE',
            'NCCL_P2P_DISABLE', 'NCCL_SHM_DISABLE', 'NCCL_NET', 'NCCL_COLLNET_ENABLE', 'GLOO_*', 'TP_SOCKET_IFNAME']


def check_env(env: dict | None) -> dict[str, str]:
    """Return the request's whitelisted env as strings; raise on anything outside the whitelist."""
    env = env or {}
    bad = [k for k in env if not any(fnmatch.fnmatchcase(k, p) for p in ENV_WHITELIST)
           or any(fnmatch.fnmatchcase(k, p) for p in ENV_DENY)]
    if bad:
        raise ValueError(f'env keys outside the whitelist: {bad}')
    out = {}
    for k, v in env.items():
        if not isinstance(v, (str, int, float, bool)):
            raise ValueError(f'env {k} must be a scalar')
        out[k] = str(int(v)) if isinstance(v, bool) else str(v)
    return out


def pick_ports(attempt_id: str, n: int = 12) -> list[int]:
    """Deterministic candidate master ports in [21000, 29000) derived from the attempt id."""
    h = int(hashlib.sha256(attempt_id.encode()).hexdigest()[:8], 16)
    return [21000 + (h + 613 * i) % 8000 for i in range(n)]


def torchrun_argv(*, nnodes: int, node_rank: int, nproc: int, master_addr: str, master_port: int,
                  request_path: str, entry: tuple[str, str] | None = None) -> list[str]:
    kind, target = entry or ('module', ENTRY_MODULE)
    tail = ['-m', target] if kind == 'module' else [target]
    return [VENV_PYTHON, '-m', 'torch.distributed.run', '--nnodes', str(nnodes), '--node_rank', str(node_rank),
            '--nproc_per_node', str(nproc), '--master_addr', master_addr, '--master_port', str(master_port),
            '--max_restarts', '0', *tail, '--request', request_path]


def node_env(*, code_root: str, request, node_rank: int) -> dict[str, str]:
    env = dict(check_env((request.config or {}).get('env')))
    env.update(SITE_ENV)                                   # immutable: site networking wins over any request value
    env['PYTHONPATH'] = ':'.join([code_root] + MCORE_STACK)
    env.update(INFRARSI_REQUEST=f'{request.out_dir}/request.json', INFRARSI_OUT_DIR=request.out_dir,
               INFRARSI_RUN_ID=request.run_id, INFRARSI_ATTEMPT_ID=request.attempt_id,
               INFRARSI_NODE_INDEX=str(node_rank), INFRARSI_NNODES=str(len(request.nodes)),
               INFRARSI_CODE_DIGEST=request.code_digest)
    if request.gpus_per_node < 8:
        env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in range(request.gpus_per_node))
    return env


def node_spec(*, request, node_rank: int, master_addr: str, master_port: int, code_root: str,
              entry: tuple[str, str] | None, deadline_s: float, stop_grace_s: float = 300,
              peer_grace_s: float = 60) -> dict:
    return {'schema': 'infrarsi-node-spec-v1', 'out_dir': request.out_dir, 'node_index': node_rank,
            'nnodes': len(request.nodes), 'node': request.nodes[node_rank],
            'argv': torchrun_argv(nnodes=len(request.nodes), node_rank=node_rank, nproc=request.gpus_per_node,
                                  master_addr=master_addr, master_port=master_port,
                                  request_path=f'{request.out_dir}/request.json', entry=entry),
            'env': node_env(code_root=code_root, request=request, node_rank=node_rank), 'cwd': request.out_dir,
            'deadline_s': deadline_s, 'stop_grace_s': stop_grace_s, 'peer_grace_s': peer_grace_s,
            'require_idle': True, 'idle_mem_mib': 2048, 'gpu_indices': None}
