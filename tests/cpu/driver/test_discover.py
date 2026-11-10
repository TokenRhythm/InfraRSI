"""workload_facts parameter counts vs checkpoint sizes; hardware probe parsers on recorded WINH output."""
from driver_helpers import HF

from infrarsi.contracts import TaskSpec
from infrarsi.discover import count_params, parse_node, workload_facts
from infrarsi.discover.hardware import parse_ibstat, parse_topo

TOPO = """\tGPU0\tGPU1\tGPU2\tNIC0\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV18\tNV18\tPIX\t0-41\t0
GPU1\tNV18\t X \tNV18\tPXB\t0-41\t0
GPU2\tNV18\tNV18\t X \tNODE\t0-41\t0
NIC0\tPIX\tPXB\tNODE\t X \t\t"""
IB = """CA 'mlx5_0'
\tCA type: MT4129
\tPort 1:
\t\tState: Active
\t\tPhysical state: LinkUp
\t\tRate: 400
\t\tLink layer: InfiniBand
CA 'mlx5_1'
\tPort 1:
\t\tState: Down
\t\tRate: 400
\t\tLink layer: InfiniBand"""


def test_param_counts_match_checkpoints():
    # task -> (expected trained params, tolerance); differences vs the checkpoint are documented parts
    exp = {'D1': 8.19e9, 'D4': 30.53e9, 'H1': 15.71e9, 'H2': 8.03e9, 'S2': 4.02e9}
    for tid, want in exp.items():
        f = workload_facts(TaskSpec.load(tid), cache_dir=HF)
        assert abs(f['params_total'] - want) / want < 0.005, (tid, f['params_total'])
        assert abs(f['params_checkpoint'] - want) / want < 0.005
    d6 = workload_facts(TaskSpec.load('D6'), cache_dir=HF)          # checkpoint also holds the MTP layer
    assert 0.97 < d6['params_total'] / d6['params_checkpoint'] < 0.99
    assert d6['layer_kinds'] == {'full': 12, 'linear': 36} and d6['moe']['num_experts'] == 512
    f1 = workload_facts(TaskSpec.load('F1'), cache_dir=HF)
    assert 120e9 < f1['params_total'] < 130e9 and abs(f1['cpu_resident_params'] - 51.2e9) < 1e8
    assert 3.0e9 < workload_facts(TaskSpec.load('D4'), cache_dir=HF)['params_active'] < 3.7e9


def test_tp_candidates_respect_heads_and_gpus():
    assert workload_facts(TaskSpec.load('D4'), cache_dir=HF)['tp_candidates'] == [1, 2, 4]   # 4 KV groups
    assert workload_facts(TaskSpec.load('D6'), cache_dir=HF)['tp_candidates'] == [1, 2]      # 2 KV groups
    assert workload_facts(TaskSpec.load('H6'), cache_dir=HF)['tp_candidates'] == [1]         # 1 GPU


def test_tied_embeddings_counted_once():
    c = {'model_type': 'qwen3', 'hidden_size': 64, 'num_hidden_layers': 2, 'vocab_size': 1000, 'num_attention_heads': 4,
         'num_key_value_heads': 2, 'head_dim': 16, 'intermediate_size': 128}
    untied, tied = count_params(c), count_params(dict(c, tie_word_embeddings=True))
    assert untied['params_total'] - tied['params_total'] == 1000 * 64


def test_parsers():
    t = parse_topo(TOPO)
    assert t['nvlink_all_pairs'] and t['nvlink_links'] == 18 and t['gpus'] == 3
    ib = parse_ibstat(IB)
    assert [c['state'] for c in ib] == ['Active', 'Down'] and ib[0]['rate_gbps'] == 400
    text = '\n'.join(['@@hostname', 'h', '@@ips', '172.27.72.155 10.0.0.2', '@@gpus',
                      '0, NVIDIA H100 80GB HBM3, 81559 MiB, 9.0, 575.57.08, 5, 16', '@@topo', TOPO, '@@ibstat', IB,
                      '@@meminfo', 'MemTotal:       1929379840 kB', 'MemAvailable:   1883063156 kB', '@@ulimit', '64',
                      '@@uid', '0', '@@df', 'gpfs01 153931627888640 80264348098560 74767279790080 52% /mnt/data',
                      '@@versions', '{"torch": "2.10.0+cu128"}', '@@end'])
    n = parse_node('WINH2', text)
    assert n['ips'][0] == '172.27.72.155' and n['gpus'][0]['memory_total_mib'] == 81559
    assert abs(n['pinned_budget_gib'] - 0.5 * 1929379840 / 2 ** 20) < 1e-6        # root: memlock does not bind
    assert n['versions']['torch'].startswith('2.10') and len(n['ib_active']) == 1
