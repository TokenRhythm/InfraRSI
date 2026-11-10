"""hls3 on ToyTrain: the original 36 tasks (GPU-hours bind, no memory declarations) - hls3 within 0.02 of hls2 in paired
frac with no more failed screens; the memory-coupled 36 tasks (toytrain.MemTask: spenders that fit only with an enabler,
a cliff near the cap, enablers that crash together, one enabler whose memory effect the simulated model does not know;
10 screens bind, as on the real arms) - hls / hls2 / hls3 / random compared, hls3 ahead of hls2 with far fewer screens
spent out of memory; and hls3 with every component off gives hls2's rows on both."""
import statistics

from infrarsi.search import toybench
from infrarsi.search.toytrain import mem_tasks, tasks


def rows_for(arms, ts, **kw):
    return {(t.name, a): toybench.run_one(t, a, **kw) for t in ts for a in arms}


def summary(rows, arms, names) -> dict:
    out = {}
    for a in arms:
        rs = [rows[(t, a)] for t in names]
        out[a] = {'frac': statistics.mean(r['frac'] for r in rs), 'median': statistics.median(r['frac'] for r in rs),
                  'oom': sum(r['oom_screens'] for r in rs), 'failed': sum(r['failed_screens'] for r in rs),
                  'probes': sum(r['probes'] for r in rs), 'gpu': statistics.mean(r['gpu_h_share'] for r in rs)}
        print(f'  {a:8s} frac mean {out[a]["frac"]:.3f} median {out[a]["median"]:.3f}  failed screens {out[a]["failed"]:3d} '
              f'(out of memory {out[a]["oom"]:3d})  probes {out[a]["probes"]:4d}  GPU-h share {out[a]["gpu"]:.2f}')
    return out


def test_toybench_hls3_keeps_up_with_hls2():
    ts = tasks(36)
    rows = rows_for(('hls2', 'hls3'), ts, budget=16)
    names = [t.name for t in ts]
    print('\noriginal toybench, 36 tasks, 16 screens of GPU-hours:')
    s = summary(rows, ('hls2', 'hls3'), names)
    d = [rows[(t, 'hls3')]['frac'] - rows[(t, 'hls2')]['frac'] for t in names]
    print(f'  paired hls3 - hls2 {statistics.mean(d):+.4f} (wins {sum(x > 1e-6 for x in d)}, losses {sum(x < -1e-6 for x in d)})')
    assert statistics.mean(d) >= -0.02 and s['hls3']['failed'] <= s['hls2']['failed']


def test_toybench_memory_coupled_scenario():
    ts = mem_tasks()
    arms = ('hls', 'hls2', 'hls3', 'random')
    rows = rows_for(arms, ts, budget=10, job_factor=None, proposer={'memory': True}, bind='screens')
    names = [t.name for t in ts]
    print('\nmemory-coupled toybench, 36 tasks, 10 screens bind (GPU-hours: 40 screens\' worth):')
    s = summary(rows, arms, names)
    d = [rows[(t, 'hls3')]['frac'] - rows[(t, 'hls2')]['frac'] for t in names]
    print(f'  paired hls3 - hls2 {statistics.mean(d):+.4f} (wins {sum(x > 1e-6 for x in d)}, losses {sum(x < -1e-6 for x in d)})')
    assert s['hls3']['frac'] >= s['hls2']['frac'] + 0.03 and s['hls3']['frac'] > s['hls']['frac'] > s['random']['frac']
    assert s['hls3']['oom'] * 4 <= s['hls2']['oom'] and sum(x > 1e-6 for x in d) > sum(x < -1e-6 for x in d)


def test_every_component_off_gives_hls2s_rows():
    keys = ('frac', 'screens', 'trials', 'probes', 'failed_screens', 'stop', 'gpu_h_share', 'curve')
    for ts, kw in ((tasks(36)[:10], {'budget': 16}),
                   (mem_tasks()[:10], {'budget': 10, 'job_factor': None, 'proposer': {'memory': True}, 'bind': 'screens'})):
        rows = rows_for(('hls2', 'hls3-off'), ts, **kw)
        for t in ts:
            a, b = rows[(t.name, 'hls2')], rows[(t.name, 'hls3-off')]
            assert {k: a[k] for k in keys} == {k: b[k] for k in keys}, t.name
