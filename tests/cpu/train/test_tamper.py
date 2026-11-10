"""In-process tamper resistance of the frozen zone against L1 overlay code (clocks, serialization, hooks)."""
import hashlib
import json
import math
import time
import types

import pytest

from infrarsi.contracts import file_digest
from infrarsi.runtime import gates, overlay_loader as l1
from infrarsi.runtime.signals import SignalWriter, dump_row

ROW = {'step': 3, 'whole_update_s': 2.5, 'fwd_bwd_s': None, 'optim_s': 0.25, 'data_s': 0.0, 'peak_alloc_gib': 33.1,
       'peak_reserved_gib': 35.0, 'alloc_retries': 0, 'alloc_ooms': 0, 'utc_start_s': 1792452534.756425,
       'utc_end_s': 1792452537.25, 'main_ce': float('nan'), 'lr': 9.99e-06, 'note_ok': True, 'microbatches': 4}


def test_dump_row_is_plain_json():
    text = dump_row(ROW)
    back = json.loads(text)
    assert set(back) == set(ROW) and back['utc_start_s'] == ROW['utc_start_s'] and math.isnan(back['main_ce'])
    assert list(back) == sorted(ROW) and back['fwd_bwd_s'] is None and back['note_ok'] is True
    with pytest.raises(TypeError):
        dump_row({'step': 1, 'nested': {'a': 1}})


def test_signal_rows_ignore_a_patched_json_module(tmp_path, monkeypatch):
    def forged(obj, *a, **k):
        return '{"whole_update_s": 0.001}'
    monkeypatch.setattr(json, 'dumps', forged)
    w = SignalWriter(tmp_path, 0)
    w.write({k: v for k, v in ROW.items() if k not in ('main_ce', 'lr', 'note_ok')})
    w.close()
    monkeypatch.undo()
    row = json.loads((tmp_path / 'signals' / 'rank0.jsonl').read_text())
    assert row['whole_update_s'] == 2.5


def test_sentinels_detect_replaced_functions(monkeypatch):
    snap = gates.sentinel_snapshot()
    assert 'time.perf_counter' in snap and 'json.encoder.JSONEncoder.encode' in snap
    assert 'infrarsi.runtime.signals.dump_row' in snap
    assert gates.sentinel_changes(snap) == []
    monkeypatch.setattr(time, 'perf_counter', lambda: 0.0)
    monkeypatch.setattr(json.encoder.JSONEncoder, 'encode', lambda self, o: '{}')
    assert set(gates.sentinel_changes(snap)) == {'time.perf_counter', 'json.encoder.JSONEncoder.encode'}
    monkeypatch.undo()
    assert gates.sentinel_changes(snap) == []


def test_instance_shadowing_detected():
    class Opt:
        def step(self):
            return True
    opt = Opt()
    assert gates.shadowed(opt, gates.OPTIMIZER_METHODS) == []
    opt.step = lambda: (True, 1.0, None)
    assert gates.shadowed(opt, gates.OPTIMIZER_METHODS) == ['step']


def test_loader_hashes_every_file_before_running_any(tmp_path):
    # host_sync.py (sorted before schedule.py) replaces hashlib.sha256 at import; schedule.py must still be
    # verified with the real hash because every file is hashed before any is executed.
    (tmp_path / 'host_sync.py').write_text(
        'import hashlib\nhashlib.sha256 = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("forged"))\n')
    (tmp_path / 'schedule.py').write_text('def forward_backward(ctx, run, **kw):\n    return run(**kw)\n')
    spec = {'dir': str(tmp_path), 'files': {'host_sync.py': file_digest(tmp_path / 'host_sync.py'),
                                            'schedule.py': file_digest(tmp_path / 'schedule.py')}}
    real = hashlib.sha256
    try:
        ov = l1.load(spec)
    finally:
        hashlib.sha256 = real
    assert ov.files['schedule.py'] == spec['files']['schedule.py'] and ov.source == 'request'


def test_payload_verification_maps_to_data_unavailable():
    from infrarsi.adapters.megatron_core.data import DataSource, DataUnavailable
    src = DataSource.__new__(DataSource)

    def bad():
        raise ValueError('tokens.bin: sha256 differs from manifest')
    src.ds = types.SimpleNamespace(verify=bad)
    with pytest.raises(DataUnavailable, match='payload differs'):
        src.verify_payload()
    src.ds = types.SimpleNamespace()          # synthetic data: nothing to verify
    src.verify_payload()


def test_checker_closure_survives_replaced_gates(monkeypatch):
    check = gates.make_checker()
    assert check() == [] and 'time.perf_counter' in check.paths
    monkeypatch.setattr(gates, 'sentinel_changes', lambda snapshot: [])     # an overlay disabling the verifier
    monkeypatch.setattr(gates, 'resolve', lambda path: None)
    monkeypatch.setattr(time, 'perf_counter', lambda: 0.0)
    assert check() == ['time.perf_counter']


def test_resolver_prefers_package_reexports(tmp_path, monkeypatch):
    pkg = tmp_path / 'reexp_pkg'
    pkg.mkdir()
    (pkg / '__init__.py').write_text('from .finalize import finalize\n')
    (pkg / 'finalize.py').write_text('def finalize():\n    return 1\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    fn = gates.resolve('reexp_pkg.finalize')
    assert callable(fn) and fn.__name__ == 'finalize' and not isinstance(fn, types.ModuleType)
    check = gates.make_checker(('reexp_pkg.finalize',))
    import reexp_pkg
    monkeypatch.setattr(reexp_pkg, 'finalize', lambda: 2)                  # only the re-export is replaced
    assert check() == ['reexp_pkg.finalize']


def test_supplied_hooks_are_guarded_before_and_after():
    ov = l1.load(None)
    seen = []
    ov.guard = seen.append
    ov.call('host_sync', 'before_update', None, 3)
    assert seen == []                               # built-in default: frozen code, nothing to check
    ov.modules = dict(ov.modules, host_sync=types.SimpleNamespace(before_update=lambda ctx, step: None))
    ov.call('host_sync', 'before_update', None, 3)
    ov.call('host_sync', 'after_update', None, 3)   # not defined by the request file -> default, unguarded
    assert seen == ['before host_sync.before_update', 'after host_sync.before_update']


def test_static_guard_runs_on_the_exact_bytes(tmp_path):
    pytest.importorskip('infrarsi.actions.patch_guard')
    (tmp_path / 'host_sync.py').write_text('from infrarsi.runtime import timers\n\ndef before_update(ctx, step):\n'
                                          '    timers._clock = lambda: 0.0\n')
    spec = {'dir': str(tmp_path), 'files': {'host_sync.py': file_digest(tmp_path / 'host_sync.py')}}
    with pytest.raises(l1.OverlayViolation, match='static guard'):
        l1.load(spec)


def test_pins_catch_replaced_holders_and_values():
    import types as _t
    leaf = _t.SimpleNamespace(optimizer=object(), is_stub_optimizer=False, config=_t.SimpleNamespace(clip_grad=1.0))
    chained = [leaf]
    check = gates.make_pins([('chained', chained, None, 'len'), ('chained[0]', chained, 0, 'is'),
                             ('leaf.optimizer', leaf, 'optimizer', 'is'), ('leaf.stub', leaf, 'is_stub_optimizer', 'eq'),
                             ('clip', leaf.config, 'clip_grad', 'eq')])
    assert check() == []
    leaf.optimizer = object()                     # a proxy holder
    leaf.is_stub_optimizer = True
    leaf.config.clip_grad = 1e9
    chained.append(leaf)
    assert check() == ['chained', 'leaf.optimizer', 'leaf.stub', 'clip']
    chained.clear()
    assert 'chained[0]' in check()


def test_value_pins_check_the_type_before_comparing():
    import types as _t

    class Liar:                                   # claims equality with everything, truthy on demand
        def __eq__(self, other):
            return True

        def __ne__(self, other):
            return False

        def __bool__(self):
            return True
    leaf = _t.SimpleNamespace(is_stub_optimizer=False, clip=1.0)
    check = gates.make_pins([('stub', leaf, 'is_stub_optimizer', 'eq'), ('clip', leaf, 'clip', 'eq')])
    leaf.is_stub_optimizer, leaf.clip = Liar(), Liar()
    assert check() == ['stub', 'clip']
