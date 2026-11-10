import math

from infrarsi.ext.optico import guards as gb
from infrarsi.ext.optico import params as pm
from infrarsi.ext.optico import parse
from infrarsi.ext.optico import workloads as wl
from optico_helpers import R3_W3_MS, R3_W3_TFLOPS, megatron_log, report_csv, report_row, torchrun_line

W3 = wl.get('W3')


def test_optico_tflops_reproduces_run_monitor_on_the_r3_log():
    text = megatron_log('W3', W3.defaults)
    assert math.isclose(parse.optico_tflops(text), 228.8333, abs_tol=1e-3)      # report.csv of r3_20261021a: 228.833
    win = parse.window_metric(parse.iterations(text))
    assert win['iter_ms'] == R3_W3_MS[2:] and math.isclose(win['mean_iter_s'], sum(R3_W3_MS[2:]) / 3000)
    assert math.isclose(win['mean_tflops'], 228.8333, abs_tol=1e-3) and 0 < win['tflops_rsd'] < 0.1
    assert parse.optico_tflops('no throughput here') == 0.0
    # fewer lines: OptiCo still averages from the third on (only iteration 3 here)
    short = megatron_log('W3', W3.defaults, n_iters=3)
    assert math.isclose(parse.optico_tflops(short), 233.6)


def test_log_facts():
    p = W3.knobs(MICRO_BATCH_SIZE=2, TP=2, OVERLAP_GRAD_REDUCE=True, AC='full', AC_LAYERS=2)
    f = parse.log_facts(megatron_log('W3', p))
    a = f['args']
    assert a['--micro-batch-size'] == '2' and a['--tensor-model-parallel-size'] == '2' and a['--sequence-parallel'] is True
    assert a['--overlap-grad-reduce'] is True and a['--recompute-num-layers'] == '2' and f['ac'] == 'full' and f['ac_layers'] == 2
    assert [r['iteration'] for r in f['iterations']] == [1, 2, 3, 4, 5] and f['iterations'][0]['lm_loss'] == 11.92976
    assert f['memory']['max_reserved_mb'] == 12668.0 and f['random_init'] and not f['oom']


def test_report_rows_with_pandas_quoting():
    rows = [report_row(W3.defaults, log_dir='1'),
            report_row(W3.knobs(MICRO_BATCH_SIZE=8), result='fail', log_dir='2',
                       reason='MICRO_BATCH_SIZE 8 caused CUDA OOM, tried 2.1 GiB (Confidence: 90)')]
    text = '# comment\n' + report_csv(rows) + '\n'
    header, got = parse.read_report(text)
    assert header == list(parse.DENSE_HEADER) and len(got) == 2
    assert got[1]['Reasons for failure'].startswith('MICRO_BATCH_SIZE 8 caused CUDA OOM, tried')
    j = parse.row_judge(got[0])
    assert j['result'] == 'success' and math.isclose(j['tflops'], 228.833, abs_tol=1e-3) and j['mem_max_mib'] == 16727
    assert pm.params_from_row(got[1])['MICRO_BATCH_SIZE'] == 8
    # monitor_gpu.sh samples and run_monitor.sh's aggregation columns
    mem = parse.memory_samples('Temp Power Memory-Usage Memory-ALL Util% \n33C 70W 4MiB 81559MiB 0%\n40C 300W 16727MiB 81559MiB 100%\n')
    assert mem['mem_max_mib'] == 16727 and mem['mem_limit_mib'] == 81559 and mem['samples'] == 2


def test_g_b1_fixed_flags():
    p = W3.knobs(MICRO_BATCH_SIZE=2, OVERLAP_GRAD_REDUCE=True, OVERLAP_PARAM_GATHER=True, FUSED_CROSS_ENTROPY=True)
    args = parse.torchrun_args(torchrun_line('W3', p))
    assert gb.fixed_flags(args, 'W3', p, ac_line='none', ac_layers_line=1).ok
    for kw, needle in [({'gbs': 512}, '--global-batch-size'), ({'seq': 2048}, '--seq-length'), ({'iters': 10}, '--train-iters'),
                       ({'precision': 'bf16'}, '--bf16 present'), ({'drop': ('--use-flash-attn',)}, '--use-flash-attn missing'),
                       ({'extra': '--fp8-format hybrid'}, '--fp8-format present')]:
        g = gb.fixed_flags(parse.torchrun_args(torchrun_line('W3', p, **kw)), 'W3', p)
        assert not g.ok and g.ids() == ['G-B1'] * len(g.rejections) and any(needle in r['detail'] for r in g.rejections), kw
    # what ran differs from what was requested
    g = gb.fixed_flags(args, 'W3', {**p, 'MICRO_BATCH_SIZE': 4})
    assert not g.ok and 'requested knobs' in g.rejections[0]['detail']
    g = gb.fixed_flags(args, 'W3', p, ac_line='sel')
    assert not g.ok and 'AC is sel' in g.rejections[0]['detail']
    # SP must follow TP > 1
    p2 = W3.knobs(TP=2)
    assert gb.fixed_flags(parse.torchrun_args(torchrun_line('W3', p2)), 'W3', p2).ok
    g = gb.fixed_flags(parse.torchrun_args(torchrun_line('W3', p2, drop=('--sequence-parallel',))), 'W3', p2)
    assert not g.ok and 'sequence-parallel' in g.rejections[0]['detail']
    assert not gb.fixed_flags({}, 'W3', p).ok                                # no torchrun line at all
    for drop, needle in [('--tp-comm-overlap', '--tp-comm-overlap missing'), ('--swiglu', '--swiglu missing'),
                         ('--group-query-attention', '--group-query-attention missing')]:
        g = gb.fixed_flags(parse.torchrun_args(torchrun_line('W3', p2, drop=(drop,))), 'W3', p2)
        assert not g.ok and needle in g.rejections[0]['detail'], drop
    w4 = wl.get('W4').defaults                                             # llama2 7B: no GQA in its script
    assert gb.fixed_flags(parse.torchrun_args(torchrun_line('W4', w4)), 'W4', w4).ok
    g = gb.fixed_flags(parse.torchrun_args(torchrun_line('W4', w4, extra='--group-query-attention')), 'W4', w4)
    assert not g.ok and 'no GQA' in g.rejections[0]['detail']
    # model shape of another workload
    w1 = wl.get('W1').defaults
    g = gb.fixed_flags(parse.torchrun_args(torchrun_line('W3', w1)), 'W1', w1)
    assert not g.ok and '--num-layers' in g.rejections[0]['detail']


def test_g_b3_completed_iterations():
    it = parse.iterations(megatron_log('W3', W3.defaults))
    assert gb.completed_iterations(it).ok
    g = gb.completed_iterations(parse.iterations(megatron_log('W3', W3.defaults, n_iters=4)))
    assert not g.ok and 'iterations [5]' in g.rejections[0]['detail']
    no_tp = parse.iterations(megatron_log('W3', W3.defaults, tflops=None))
    g = gb.completed_iterations(no_tp)
    assert not g.ok and any('no throughput value for iterations [3, 4, 5]' in r['detail'] for r in g.rejections)
    g = gb.completed_iterations(parse.iterations(megatron_log('W3', W3.defaults, total=10)))
    assert not g.ok and 'total iterations' in g.rejections[0]['detail']


def test_g_b2_report_vs_veto_and_enabling_rule():
    assert gb.loss1(parse.iterations(megatron_log('W3', W3.defaults))) == 11.92976
    soft = gb.GB2Policy(reference_loss1=11.93, hard=False)
    hard = gb.GB2Policy(reference_loss1=11.93, hard=True)
    g = gb.iteration1_loss(12.5, soft)
    assert g.ok and g.reports[0]['exceeded'] is True and g.reports[0]['veto'] is False
    g = gb.iteration1_loss(12.5, hard)
    assert not g.ok and g.ids() == ['G-B2']
    assert gb.iteration1_loss(11.95, hard).ok
    g = gb.iteration1_loss(None, hard)
    assert g.ok and g.reports[0]['exceeded'] is None                        # no value: no check, never a veto
    on, why = gb.enabling_rule(11.93, 11.95)
    assert on and 'hard veto' in why
    off, why = gb.enabling_rule(11.93, 12.5)
    assert not off and 'stays a report' in why
    assert gb.enabling_rule(None, 11.9)[0] is False
    pol = gb.policy_from_receipts({'loss': {'iter1': 11.93}, 'attempt_id': 'a1'}, {'loss': {'iter1': 11.94}})
    assert pol.hard and pol.reference_loss1 == 11.93 and pol.reference_attempt == 'a1'
