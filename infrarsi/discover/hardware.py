"""hardware_profile(nodes): read-only probe of GPUs, NVLink topology, IB, CPU, memory, disk and library versions.

The summary block is what the planner consumes; default_profile() gives the same block for the WINH H100 fleet
without SSH (CPU tests, offline planning). pinned_budget_gib is a planning policy: half of host RAM unless
RLIMIT_MEMLOCK binds a non-root user (root holds CAP_IPC_LOCK, so the container's 64 KiB limit does not apply).
"""
from __future__ import annotations

import json
import re
import shlex

from infrarsi.contracts import digest, utc_now
from infrarsi.harness.launch import MCORE_STACK, VENV_PYTHON

PINNED_FRACTION = 0.5
H100_MEM_GIB = 81559 / 1024

VERSIONS_PY = r'''
import importlib.metadata as m, json, sys
out = {'python': sys.version.split()[0]}
for dist in ('torch', 'transformer_engine', 'transformer-engine', 'triton', 'flash_attn', 'flash-attn',
             'flash-linear-attention', 'fla', 'modelscope', 'transformers', 'datasets', 'numpy', 'deep_ep'):
    try: out[dist] = m.version(dist)
    except Exception: pass
try:
    import torch
    out['cuda'] = torch.version.cuda
    out['nccl'] = '.'.join(map(str, torch.cuda.nccl.version()))
    out['cudnn'] = torch.backends.cudnn.version()
except Exception as e: out['torch_error'] = repr(e)
try:
    from megatron.core import package_info
    out['megatron_core'] = package_info.__version__
except Exception as e: out['megatron_core_error'] = repr(e)
try:
    import mcore_bridge
    out['mcore_bridge'] = getattr(mcore_bridge, '__version__', 'present')
except Exception as e: out['mcore_bridge_error'] = repr(e)
print(json.dumps(out))
'''


def probe_script() -> str:
    py = shlex.quote(VERSIONS_PY)
    pp = ':'.join(MCORE_STACK)
    return '\n'.join([
        "echo '@@hostname'; hostname", "echo '@@ips'; hostname -I",
        "echo '@@gpus'; nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version,pcie.link.gen.max,"
        "pcie.link.width.max --format=csv,noheader",
        "echo '@@smiq'; nvidia-smi -q | grep -E 'Driver Version|CUDA Version|Product Architecture|MIG Mode|Current  *:|Persistence Mode' | head -20",
        "echo '@@topo'; nvidia-smi topo -m", "echo '@@ibstat'; ibstat 2>/dev/null", "echo '@@lscpu'; lscpu",
        "echo '@@meminfo'; grep -E 'MemTotal|MemAvailable' /proc/meminfo", "echo '@@ulimit'; ulimit -l", "echo '@@uid'; id -u",
        "echo '@@df'; df -B1 /data | tail -1",
        f"echo '@@versions'; cd /tmp && PYTHONPATH={pp} timeout 120 {VENV_PYTHON} -c {py} 2>/dev/null || echo '{{}}'",
        "echo '@@end'"])


def _sections(text: str) -> dict[str, str]:
    out, cur = {}, None
    for line in text.splitlines():
        m = re.fullmatch(r'@@(\w+)', line.strip())
        if m:
            cur = m.group(1)
            out[cur] = []
        elif cur:
            out[cur].append(line)
    return {k: '\n'.join(v) for k, v in out.items()}


def parse_topo(text: str) -> dict:
    rows = [r for r in text.splitlines() if r.strip()]
    if not rows:
        return {'gpu_pairs': {}, 'nvlink_all_pairs': False, 'nvlink_links': None}
    header = re.sub(r'\x1b\[[0-9;]*m', '', rows[0]).split()
    gpus = [h for h in header if re.fullmatch(r'GPU\d+', h)]
    pairs, links = {}, set()
    for r in rows[1:]:
        cells = re.sub(r'\x1b\[[0-9;]*m', '', r).split()
        if not cells or not re.fullmatch(r'GPU\d+', cells[0]):
            continue
        for j, g in enumerate(gpus):
            if j + 1 < len(cells) and g != cells[0]:
                pairs[f'{cells[0]}-{g}'] = cells[j + 1]
                m = re.fullmatch(r'NV(\d+)', cells[j + 1])
                if m:
                    links.add(int(m.group(1)))
    nv = bool(pairs) and all(v.startswith('NV') for v in pairs.values())
    return {'gpus': len(gpus), 'nvlink_all_pairs': nv, 'nvlink_links': min(links) if links else None,
            'kinds': sorted(set(pairs.values()))}


def parse_ibstat(text: str) -> list[dict]:
    cas, cur = [], None
    for line in text.splitlines():
        m = re.match(r"CA '(\S+)'", line.strip())
        if m:
            cur = {'name': m.group(1)}
            cas.append(cur)
            continue
        if cur is None:
            continue
        for key, field in (('State:', 'state'), ('Rate:', 'rate_gbps'), ('Link layer:', 'link_layer'), ('Physical state:', 'phys')):
            if line.strip().startswith(key):
                v = line.split(':', 1)[1].strip()
                cur[field] = int(v) if field == 'rate_gbps' and v.isdigit() else v
    return cas


def parse_lscpu(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            kv[k.strip()] = v.strip()
    def num(k):
        try:
            return int(kv.get(k, ''))
        except ValueError:
            return None
    return {'model': kv.get('Model name'), 'sockets': num('Socket(s)'), 'cores_per_socket': num('Core(s) per socket'),
            'threads_per_core': num('Thread(s) per core'), 'cpus': num('CPU(s)'), 'numa_nodes': num('NUMA node(s)'),
            'online': kv.get('On-line CPU(s) list')}


def parse_node(node: str, text: str) -> dict:
    s = _sections(text)
    gpus = []
    for line in s.get('gpus', '').splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) >= 7:
            gpus.append({'index': int(parts[0]), 'name': parts[1], 'memory_total_mib': int(parts[2].split()[0]),
                         'compute_cap': parts[3], 'driver': parts[4], 'pcie_gen': parts[5], 'pcie_width': parts[6]})
    mem = {}
    for line in s.get('meminfo', '').splitlines():
        k, v = line.split(':', 1)
        mem[k.strip()] = int(v.split()[0]) * 1024
    ib = parse_ibstat(s.get('ibstat', ''))
    df = s.get('df', '').split()
    try:
        versions = json.loads(s.get('versions', '{}').strip().splitlines()[-1])
    except (ValueError, IndexError):
        versions = {}
    ul = s.get('ulimit', '').strip()
    uid = s.get('uid', '').strip()
    host_gib = mem.get('MemTotal', 0) / 2 ** 30
    memlock = None if ul in ('unlimited', '') else int(ul) * 1024 / 2 ** 30
    binds = memlock is not None and uid != '0'
    return {'node': node, 'hostname': s.get('hostname', '').strip(), 'ips': s.get('ips', '').split(), 'gpus': gpus,
            'smi': s.get('smiq', '').strip().splitlines(), 'topology': parse_topo(s.get('topo', '')), 'ib': ib,
            'ib_active': [c for c in ib if c.get('state') == 'Active' and c.get('link_layer') == 'InfiniBand'],
            'cpu': parse_lscpu(s.get('lscpu', '')), 'host_mem_gib': host_gib,
            'host_mem_available_gib': mem.get('MemAvailable', 0) / 2 ** 30, 'memlock_gib': memlock, 'uid': uid,
            'pinned_budget_gib': min(memlock, host_gib) if binds else PINNED_FRACTION * host_gib,
            'data_free_bytes': int(df[3]) if len(df) > 3 and df[3].isdigit() else None, 'versions': versions}


def summarize(nodes: list[dict]) -> dict:
    g0 = nodes[0]['gpus'][0] if nodes and nodes[0]['gpus'] else {}
    mem_gib = g0.get('memory_total_mib', 0) / 1024
    name = g0.get('name', '')
    gtype = 'H100-80GB' if 'H100' in name and mem_gib > 70 else ('H20-96GB' if 'H20' in name else
                                                                 ('L20X-141GB' if 'L20' in name else name))
    ib = [c for n in nodes for c in n['ib_active']]
    return {'nodes': len(nodes), 'gpus_per_node': min(len(n['gpus']) for n in nodes) if nodes else 0, 'gpu_name': name,
            'gpu_type': gtype, 'gpu_mem_gib': mem_gib, 'compute_cap': g0.get('compute_cap'),
            'nvlink_intra': all(n['topology'].get('nvlink_all_pairs') for n in nodes),
            'nvlink_links': min((n['topology'].get('nvlink_links') or 0) for n in nodes) if nodes else None,
            'ib_ports_active_per_node': min(len(n['ib_active']) for n in nodes) if nodes else 0,
            'ib_rate_gbps': max((c.get('rate_gbps') or 0) for c in ib) if ib else 0,
            'ib_inter': all(len(n['ib_active']) > 0 for n in nodes),
            'host_mem_gib': min(n['host_mem_gib'] for n in nodes) if nodes else 0,
            'pinned_budget_gib': min(n['pinned_budget_gib'] for n in nodes) if nodes else 0,
            'cpus_per_node': min((n['cpu'].get('cpus') or 0) for n in nodes) if nodes else 0,
            'peak_bf16_tflops': 989.0 if gtype == 'H100-80GB' else (148.0 if gtype == 'H20-96GB' else None)}


def hardware_profile(nodes: list[str], remote=None) -> dict:
    """Read-only: one combined probe per node (sequential, to stay gentle on sshd)."""
    from infrarsi.harness.remote import default_remote
    remote = remote or default_remote()
    parsed = []
    for n in nodes:
        res = remote.sh(n, probe_script(), timeout=300)
        parsed.append(parse_node(n, res['stdout']))
    prof = {'schema': 'infrarsi-hardware-profile-v1', 'created_utc': utc_now(), 'nodes': parsed, 'summary': summarize(parsed)}
    prof['digest'] = digest({'summary': prof['summary'], 'versions': [p['versions'] for p in parsed]})
    return prof


def default_profile(nodes: int = 1, gpus_per_node: int = 8) -> dict:
    """WINH fleet facts verified 2026-10-20 (8 x H100 80GB HBM3 per node, NV18, 11 x mlx5 400G, 1,840 GiB RAM)."""
    host = 1929379840 * 1024 / 2 ** 30
    return {'schema': 'infrarsi-hardware-profile-v1', 'nodes': [], 'source': 'default WINH H100 profile',
            'summary': {'nodes': nodes, 'gpus_per_node': gpus_per_node, 'gpu_name': 'NVIDIA H100 80GB HBM3',
                        'gpu_type': 'H100-80GB', 'gpu_mem_gib': H100_MEM_GIB, 'compute_cap': '9.0', 'nvlink_intra': True,
                        'nvlink_links': 18, 'ib_ports_active_per_node': 11, 'ib_rate_gbps': 400, 'ib_inter': True,
                        'host_mem_gib': host, 'pinned_budget_gib': PINNED_FRACTION * host, 'cpus_per_node': 168,
                        'peak_bf16_tflops': 989.0}}
