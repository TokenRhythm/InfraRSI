"""C3 - the log scanner and the evidence block: real-world log formats, a fuzz of key/value spellings against a known
truth, the failure-event scanner, the clock rules (bursts, stalls, split halves) and the probe reader."""
import json
import random
import time

import pytest

from infrarsi.project import signals as sig

# ---------------------------------------------------------------- real-world formats
FORMATS = {
    'megatron': (' iteration        7/      30 | consumed samples:          448 | elapsed time per iteration (ms): 1234.5 | '
                 'learning rate: 1.000000E-05 | global batch size:    64 | lm loss: 2.345678E+00 | loss scale: 1.0 | grad norm: 1.234 | '
                 'number of zeros: 0 |', {'step': 7, 'loss': 2.345678, 'grad_norm': 1.234, 'lr': 1e-5, 'samples': 64, 'step_time': 1.2345}),
    'torchtitan': ('step:  7  loss:  2.3100  memory: 12.34GiB(15.6%)  tps: 1234  mfu: 38.2%', {'step': 7, 'loss': 2.31, 'mem_gib': 12.34}),
    'nanogpt': ('iter 7: loss 2.2801, time 123.45ms, mfu 38.20%', {'step': 7, 'loss': 2.2801}),
    'hf_dict': ("{'loss': 2.3, 'grad_norm': 1.1, 'learning_rate': 5e-05, 'epoch': 0.1}", {'step': 1, 'loss': 2.3, 'grad_norm': 1.1, 'lr': 5e-5}),
    'kv_equals': ('step=7 loss=2.25 lr=1e-4 tokens_per_step=8192 tokens/s=1234 data_digest=deadbeefcafe', {'step': 7, 'loss': 2.25, 'tokens': 8192, 'data': 'deadbeefcafe'}),
    'timestamp_prefix': ('[2026-10-25 10:00:00,123] [INFO] [train] step 7 | loss 2.25 | grad_norm 0.9', {'step': 7, 'loss': 2.25, 'grad_norm': 0.9}),
    'rank_prefix': ('[rank0]: step 7 loss 2.25 lr 1e-4', {'step': 7, 'loss': 2.25, 'lr': 1e-4}),
    'bracket_step': ('Step [7/30] loss: 2.25 lr: 1.0e-04', {'step': 7, 'loss': 2.25, 'lr': 1e-4}),
    'global_step': ('global_step=7 train/loss=2.25 train/grad_norm=0.9', {'step': 7, 'loss': 2.25, 'grad_norm': 0.9}),
    'scientific': ('step 7 loss 2.25e+00 grad_norm 9.0E-01 lr 1E-4', {'step': 7, 'loss': 2.25, 'grad_norm': 0.9, 'lr': 1e-4}),
    'nan_inf': ('step 7 loss nan grad_norm inf', {'step': 7, 'loss': 'nan', 'grad_norm': 'inf'}),
    'memory_mib': ('step 7 loss 2.25 max memory reserved 20480MiB', {'step': 7, 'loss': 2.25, 'mem_gib': 20.0}),
    'memory_gb': ('step 7 loss 2.25 peak memory 21.47GB', {'step': 7, 'loss': 2.25, 'mem_gib': pytest.approx(20.0, abs=0.01)}),
    'loss_scale_is_not_loss': ('step 7 loss_scale 65536 lm loss 2.25', {'step': 7, 'loss': 2.25}),
    'deepspeed': ('[2026-10-25 10:00:00,123] [INFO] [logging.py:96:log_dist] [Rank 0] step=7, skipped=0, lr=[1e-05], mom=[(0.9, 0.95)]',
                  {'step': 7, 'lr': 1e-5}),
    'ansi': ('\x1b[32mstep 7\x1b[0m loss \x1b[1m2.25\x1b[0m', {'step': 7, 'loss': 2.25}),
}
PARSER_KW = {'hf_dict': {'count_steps': True}}        # HF Trainer's dicts carry no step: `init` sets count_steps for them
# formats the scanner does not read, kept as strict expected failures so a fix shows up (F-SYS-3 / 6 / 7 were fixed)
FINDINGS = {}


@pytest.mark.parametrize('name', [pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=FINDINGS[n])) if n in FINDINGS else n
                                  for n in sorted(FORMATS)])
def test_known_log_format(name):
    line, want = FORMATS[name]
    rows = sig.LogParser(**PARSER_KW.get(name, {})).parse([(1.0, line)])
    assert rows, f'{name}: no row from {line!r}'
    row = rows[0]
    for k, v in want.items():
        assert k in row, f'{name}: {k} missing in {row}'
        assert row[k] == v, f'{name}: {k} = {row[k]!r}, want {v!r}'


def test_progress_bars_are_read_through_the_drafted_patterns():
    """A tqdm bar is not a log line the scanner reads by itself (its contract: `init` drafts patterns for diffusers'
    scripts, D1-05 / D2-R205-2); with them, each count's last redraw holds the values and the 0/N line is no update."""
    from infrarsi.project.detect import TQDM_STEPS
    bare = sig.LogParser().parse([(1.0, 'Steps:  23%|##3       | 7/30 [00:03<00:29, 1.09s/it, lr=1e-5, step_loss=2.25]')])
    assert bare == []
    lines = [(0.5, 'Steps:   0%|          | 0/30 [00:00<?, ?it/s]'),
             (1.0, 'Steps:   3%|          | 1/30 [00:01<00:29, 1.00s/it]'),
             (1.1, 'Steps:   3%|          | 1/30 [00:01<00:29, 1.00s/it, lr=1e-5, step_loss=2.30]'),
             (2.0, 'Steps:   7%|          | 2/30 [00:02<00:28, 1.00s/it, lr=1e-5, step_loss=2.30]'),
             (2.1, 'Steps:   7%|          | 2/30 [00:02<00:28, 1.00s/it, lr=1e-5, step_loss=2.25]')]
    rows = sig.LogParser(patterns=dict(TQDM_STEPS)).parse(lines)
    assert [r['step'] for r in rows] == [1, 2] and rows[0]['loss'] == 2.30 and rows[1]['loss'] == 2.25
    assert rows[0]['t'] == 1.0 and rows[1]['t'] == 2.0              # the first redraw of a count gives the time


def test_validation_and_eval_lines_are_ignored_by_default():
    lines = [(1.0, 'step 1 loss 2.0'), (2.0, 'validation loss at step 1: 9.9'), (3.0, 'eval loss 8.8 step 1'),
             (4.0, 'test loss: 7.7'), (5.0, 'step 2 loss 1.9')]
    rows = sig.LogParser(ignore=r'(?i)\b(?:val(?:idation)?|eval(?:uation)?|test)\b').parse(lines)
    assert [r['step'] for r in rows] == [1, 2] and all(r['loss'] < 3 for r in rows)


def test_explicit_patterns_override_the_scanner():
    p = sig.LogParser(patterns={'loss': r'L=(?P<v>[\d.]+)', 'step': r'it (?P<v>\d+)'})
    rows = p.parse([(1.0, 'it 3 L=1.5 loss 9.9')])
    assert rows[0]['step'] == 3 and rows[0]['loss'] == 1.5


def test_count_steps_numbers_unnumbered_lines():
    rows = sig.LogParser(count_steps=True).parse([(1.0, "{'loss': 2.3}"), (2.0, "{'loss': 2.2}"), (3.0, 'other line')])
    assert [r['step'] for r in rows] == [1, 2]


# ---------------------------------------------------------------- fuzz: spellings against a truth
SEPS = [': ', '=', ' ', ':', ' = ']
KEYS = {'loss': ['loss', 'lm loss', 'train_loss', 'training loss', 'train/loss'], 'grad_norm': ['grad norm', 'grad_norm', 'gnorm'],
        'lr': ['lr', 'learning rate', 'learning_rate'], 'tokens': ['tokens', 'tokens_per_step', 'ntokens']}


def _fuzz_line(rng):
    truth = {'step': rng.randint(1, 5000), 'loss': round(rng.uniform(0.1, 9), 4), 'grad_norm': round(rng.uniform(0.1, 9), 3),
             'lr': float(f'{rng.uniform(1e-6, 1e-2):.3e}'), 'tokens': rng.randint(1, 10 ** 6)}
    parts = [f'{rng.choice(["step", "iter", "iteration", "global_step"])}{rng.choice(SEPS)}{truth["step"]}']
    for k in rng.sample(list(KEYS), len(KEYS)):
        v = truth[k]
        text = f'{v:.3e}' if k == 'lr' else str(v)
        parts.append(f'{rng.choice(KEYS[k])}{rng.choice(SEPS)}{text}')
    noise = [f'epoch{rng.choice(SEPS)}{rng.randint(0, 3)}', f'mfu{rng.choice(SEPS)}{rng.uniform(1, 60):.1f}%', f'time{rng.choice(SEPS)}{rng.uniform(1, 999):.1f}ms']
    parts += rng.sample(noise, rng.randint(0, 3))
    rng.shuffle(parts[1:])
    joiner = rng.choice([' | ', ', ', '  ', ' '])
    return joiner.join(parts), truth


def test_fuzzed_spellings_parse_to_the_truth():
    rng = random.Random(11)
    bad = []
    for i in range(500):
        line, truth = _fuzz_line(rng)
        rows = sig.LogParser().parse([(1.0, line)])
        row = rows[0] if rows else {}
        for k, v in truth.items():
            got = row.get(k)
            ok = got == v or (isinstance(got, float) and isinstance(v, (int, float)) and abs(got - v) <= 1e-9 * max(1, abs(v)))
            if not ok:
                bad.append((line, k, got, v))
    assert not bad, f'{len(bad)} of 500 lines mis-parsed, e.g. {bad[:5]}'


# ---------------------------------------------------------------- failure events
EVENTS = {
    'oom': ('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB', {'oom': 1, 'error': 'torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB'}),
    'nccl': ('[rank3]: NCCL error: ncclSystemError: System call failed', {'collective': 1}),
    'watchdog': ('[rank0]:[E ProcessGroupNCCL.cpp:563] Watchdog caught collective operation timeout', {'collective': 1}),
    'traceback': ('Traceback (most recent call last):', {'traceback': 1}),
    'rank_traceback': ('[rank2]: Traceback (most recent call last):', {'traceback': 1}),
    'exception_line': ('RuntimeError: CUDA error: an illegal memory access was encountered', {'error': 'RuntimeError: CUDA error: an illegal memory access was encountered'}),
    'module_exception': ('torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address', {'error': 'torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address'}),
    'argparse': ('train.py: error: argument --mode: invalid choice: \'x\'', {'error': 'train.py: error: argument --mode: invalid choice: \'x\''}),
    'no_such_file': ('run.sh: line 18: /x/bin/python\r: No such file or directory', {'error': 'run.sh: line 18: /x/bin/python\r: No such file or directory'}),
    'command_not_found': ('bash: torchrun: command not found', {'error': 'bash: torchrun: command not found'}),
    'launcher_summary_is_not_the_error': ('torch.distributed.elastic.multiprocessing.errors.ChildFailedError:', {}),
    'plain_text': ('the model is loading', {}),
}


@pytest.mark.parametrize('name', sorted(EVENTS))
def test_event_lines(name):
    line, want = EVENTS[name]
    ev = sig.scan_events([(1.0, line)])
    for k, v in want.items():
        assert ev.get(k) == v, (name, k, ev)
    if 'error' not in want:
        assert 'error' not in ev
    for k in ('oom', 'collective', 'traceback'):
        if k not in want:
            assert ev[k] == 0


def test_the_first_error_line_wins_and_samples_are_capped():
    lines = [(1.0, 'ValueError: first'), (2.0, 'RuntimeError: second')] + [(3.0 + i, 'NCCL error: x') for i in range(9)]
    ev = sig.scan_events(lines)
    assert ev['error'] == 'ValueError: first' and ev['collective'] == 9 and len(ev['samples']) == 5


# ---------------------------------------------------------------- the clock
def test_clock_regular_steps_and_rsd():
    rows = [{'step': s, 't': 100.0 + 0.5 * s} for s in range(1, 13)]
    t, rsd, note = sig.clock_time(rows, [5, 12])
    assert t == pytest.approx(0.5) and rsd == pytest.approx(0.0, abs=1e-9) and note is None


def test_clock_bursts_give_no_time():
    """Buffered stdout: clumps of lines separated by real gaps (the median interval tiny, the mean not)."""
    rows = [{'step': s, 't': 100.0 + (0.5 * (s // 4)) + 1e-6 * s} for s in range(1, 13)]
    t, _, note = sig.clock_time(rows, [5, 12])
    assert t is None and 'burst' in note


def test_a_log_that_arrives_all_at_once_is_not_a_microsecond_step_time():
    """F-SYS-4: every interval a few microseconds is no training job; it is buffered output (fixed: the median interval
    below a millisecond is bursts, whatever the mean)."""
    rows = [{'step': s, 't': 100.0 + 1e-6 * s} for s in range(1, 13)]
    t, _, note = sig.clock_time(rows, [5, 12])
    assert t is None and note


def test_one_stall_moves_the_mean_less_on_the_robust_rsd():
    rows = [{'step': s, 't': 100.0 + 0.5 * s + (3.0 if s >= 8 else 0.0)} for s in range(1, 13)]
    _, rsd, _ = sig.clock_time(rows, [5, 12])
    robust = sig.robust_rsd(sig.intervals(rows, [5, 12])[1])
    assert robust < rsd
    half = sig.split_half(rows, [5, 12])
    assert half is not None and half < 0                          # the second half is slower


def test_evidence_fields_from_rows():
    rows = [{'step': s, 't': 10.0 + s, 'loss': 2.0 - 0.01 * s, 'tokens': 4096, 'mem_gib': 5.0 + s} for s in range(0, 12)]
    ev = sig.evidence(rows, steps=12, window=[5, 12], exit_code=0, timed_out=False, events={}, frozen={}, meta={'trainable_params': 3})
    assert ev['steps_completed'] == 12 and ev['step_base'] == 0 and ev['logged_every_step'] and ev['memory_peak_gib'] == 16.0
    assert ev['step_time'] == pytest.approx(1.0) and ev['step_time_source'] == 'clock' and ev['meta'] == {'trainable_params': 3}
    gaps = [r for r in rows if r['step'] not in (3, 4)]
    ev2 = sig.evidence(gaps, steps=12, window=[5, 12], exit_code=0, timed_out=False, events={}, frozen={})
    assert not ev2['logged_every_step'] and ev2['steps_completed'] == 12
    ev3 = sig.evidence([], steps=12, window=[5, 12], exit_code=0, timed_out=False, events={}, frozen={})
    assert ev3['steps_completed'] == 0 and ev3['step_time'] is None


def test_reported_step_time_when_the_clock_fails():
    rows = [{'step': s, 't': 100.0 + (0.5 * (s // 4)) + 1e-6 * s, 'step_time': 0.25} for s in range(1, 13)]
    ev = sig.evidence(rows, steps=12, window=[5, 12], exit_code=0, timed_out=False, events={}, frozen={})
    assert ev['step_time'] == 0.25 and ev['step_time_source'] == 'reported' and ev['clock_note']


def test_probe_reader():
    text = '\n'.join([json.dumps({'meta': {'trainable_params': 10, 'precision': {'param': 'bf16'}}}),
                      json.dumps({'step': 1, 't': 1.0, 'loss': 2.0}), json.dumps({'step': 1, 't': 1.5, 'loss': 9.0}),
                      'not json', json.dumps({'step': 2, 't': 2.0, 'loss': float('nan')}), json.dumps([1, 2])])
    rows, meta = sig.read_probe(text)
    assert [r['step'] for r in rows] == [1, 2] and rows[0]['loss'] == 2.0 and rows[1]['loss'] == 'nan'
    assert meta == {'trainable_params': 10, 'precision': {'param': 'bf16'}}


def test_megatron_timer_block_gives_phases():
    lines = [(1.0, ' iteration        7/      30 | elapsed time per iteration (ms): 1000.0 | lm loss: 2.0 |'),
             (1.1, '    forward-backward ...........................: 800.00'),
             (1.2, '    forward-compute ............................: 300.00'),
             (1.3, '    optimizer ..................................: 150.00')]
    rows = sig.LogParser(megatron_timers=True).parse(lines)
    assert rows and rows[0]['phases']['forward-backward'] == 800.0 and rows[0]['phases']['optimizer'] == 150.0


def test_scanner_throughput():
    lines = [(float(i), f'step {i} | loss {2.0 - i * 1e-6:.6f} | grad_norm 1.1 | lr 1e-4 | tokens 4096 | mem 12.3GiB') for i in range(1, 20001)]
    t0 = time.perf_counter()
    rows = sig.LogParser().parse(lines)
    dt = time.perf_counter() - t0
    assert len(rows) == 20000
    assert dt < 20, f'20k lines took {dt:.1f} s'                      # a loose bound: a regression shows as minutes
