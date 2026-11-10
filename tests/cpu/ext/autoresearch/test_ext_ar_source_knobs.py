import ast
import py_compile

import pytest

from infrarsi.ext.autoresearch import infra_only_check, judge_files, knobs, logs, source
from x1_ar_helpers import train_log


def test_base_tree_is_upstream_plus_d_ar_2():
    base = source.base_train()
    assert source.sha256_text(base) == source.BASE_TRAIN_SHA256
    up = source.upstream('train.py')
    assert [l for l in base.splitlines() if l not in up.splitlines()] == [source.D_AR_2_NEW]
    assert source.KERNEL_REV == 'cffde39aa8c34740f53e8d04a645ef8beefbccc9'           # the R2 snapshot directory name
    assert source.node_path('data/shard_00000.parquet', '/c') == f'{source.NODE_DATA}/shard_00000.parquet'
    assert source.node_path('repo/prepare.py', '/c') == '/c/repo/prepare.py'
    with pytest.raises(ValueError):
        source.node_path('etc/passwd', '/c')


def test_model_ref_is_verbatim_baseline_model():
    ref = judge_files.model_ref_source()
    tree = ast.parse(ref)
    names = [n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    assert names == list(judge_files.MODEL_NAMES) + ['build_model_config', 'expected_config']
    base = source.base_train()
    for cls in ('class GPT(nn.Module):', 'class CausalSelfAttention(nn.Module):', '@dataclass\nclass GPTConfig:'):
        assert cls in ref and cls in base
    seg = base[base.index('class GPT(nn.Module):'):base.index('# Optimizer (MuonAdamW')]
    assert seg.strip().rsplit('\n', 1)[0] in ref                       # the whole GPT class, byte for byte
    assert 'fa3 = get_kernel("kernels-community/flash-attn3").flash_attn_interface' in ref and 'from prepare import MAX_SEQ_LEN' in ref
    assert 'DEPTH = 8' in ref and 'ASPECT_RATIO = 64' in ref and 'MuonAdamW(' in ref    # setup_optimizer kept, never called


def test_trusted_files_compile(tmp_path):
    files = judge_files.trusted_files()
    assert set(files) == {'judge/model_ref.py', 'judge/judge_bpb.py', 'judge/prepare.py', 'judge/base_train.py', 'tools/run_train.py',
                          'tools/code_attest.py', 'repo/prepare.py'}
    assert files['judge/base_train.py'] == source.base_train() and 'import torch' not in files['tools/code_attest.py']
    for name, text in files.items():
        p = tmp_path / name.replace('/', '_')
        p.write_text(text, encoding='utf-8')
        py_compile.compile(str(p), doraise=True)
    assert files['judge/prepare.py'] == source.upstream('prepare.py')
    js = files['judge/judge_bpb.py']
    assert 'weights_only=True' in js and 'strict=True' in js and 'evaluate_bpb' in js and 'torch.compile(model, dynamic=False)' in js
    rt = files['tools/run_train.py']
    assert "runpy.run_path(train_path, run_name='__main__')" in rt and '_orig_mod' in rt
    assert rt.index('import code_attest') < rt.index('sys.path.insert(0, repo)') and 'code_attest.live(' in rt
    assert judge_files.trusted_sha256() == {k: source.sha256_text(v) for k, v in files.items()}


def test_knob_space_and_rendering():
    pts = knobs.points()
    assert len(pts) == 48 == len({tuple(p.values()) for p in pts}) and knobs.DEFAULTS in pts      # AMENDMENT_EXT_007
    order = knobs.sample_order(42)
    assert len(order) == 47 and knobs.DEFAULTS not in order and order == knobs.sample_order(42) != knobs.sample_order(43)
    base = source.base_train()
    assert knobs.apply(base, knobs.DEFAULTS) == base and knobs.apply(base, {}) == base
    t = knobs.apply(base, {'DEVICE_BATCH_SIZE': 64, 'compile_mode': 'max-autotune-no-cudagraphs', 'compile_dynamic': None,
                           'expandable_segments': False, 'optimizer_foreach': True})
    assert 'DEVICE_BATCH_SIZE = 64  # per-device batch size' in t
    assert "model = torch.compile(model, dynamic=None, mode='max-autotune-no-cudagraphs')" in t
    assert 'expandable_segments:False' in t and 'torch._foreach_lerp_(exp_avgs, grads, 1 - beta1)' in t
    assert t.count('def _step_adamw') == 1 and "adamw_step_fused(p, grad, state['exp_avg']" not in t
    for bad in ({'DEVICE_BATCH_SIZE': 96}, {'compile_dynamic': 0}, {'optimizer_foreach': 1}, {'unknown': 1},
                {'compile_mode': 'reduce-overhead'}, {'compile_mode': 'max-autotune'}, {'DEVICE_BATCH_SIZE': 256}):
        with pytest.raises(knobs.KnobError):              # the removed values are outside the space (EXT_007)
            knobs.apply(base, bad)
    with pytest.raises(knobs.KnobError):
        knobs.apply(base.replace(knobs.COMPILE_LINE, 'model = model'), {'compile_mode': 'max-autotune-no-cudagraphs'})
    for p in order:                                       # every point keeps the infra-only contract by construction
        assert infra_only_check.static(base, knobs.apply(base, p)).ok


def test_frozen_knob_space_c():
    """AMENDMENT_EXT_007: the pruned space, the removed values with their evidence and the fresh seed-42 order are
    frozen in knob_space_C.json; the code refuses a file that drifted; its hash reaches every C-RND task."""
    import hashlib
    import json
    from pathlib import Path
    path = Path(knobs.__file__).resolve().parent / knobs.KNOB_SPACE_FILE
    raw = path.read_bytes()
    data = json.loads(raw)
    assert data == json.loads(json.dumps(knobs.frozen_space()))
    assert data['size'] == 48 and data['size_without_baseline'] == 47 and data['seed'] == 42
    assert {(r['knob'], r['value'], r['error_class']) for r in data['removed']} == {
        ('compile_mode', 'reduce-overhead', 'cuda_graph'), ('compile_mode', 'max-autotune', 'cuda_graph'),
        ('DEVICE_BATCH_SIZE', 256, 'oom')}
    assert all(r['evidence'] for r in data['removed'])
    assert data['draw_order'] == json.loads(json.dumps(knobs.sample_order(42)))
    ref = knobs.frozen_reference()
    assert ref['sha256'] == hashlib.sha256(raw).hexdigest() and ref['size'] == 48


def test_frozen_knob_space_drift_is_refused(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    data = json.loads((Path(knobs.__file__).resolve().parent / knobs.KNOB_SPACE_FILE).read_text(encoding='utf-8'))
    data['draw_order'] = list(reversed(data['draw_order']))
    p = tmp_path / 'knob_space_C.json'
    p.write_text(json.dumps(data), encoding='utf-8')
    monkeypatch.setattr(knobs, '_space_path', lambda: p)
    with pytest.raises(knobs.KnobError, match='does not match'):
        knobs.frozen_reference()


def test_log_parsers():
    log = train_log()
    s = logs.summary(log)
    assert s['val_bpb'] == 0.9977 and s['num_steps'] == 948 and s['training_seconds'] == 300.1 and s['depth'] == 8
    rows = logs.steps(log)
    assert rows[0]['dt_ms'] == 31937 and rows[1]['tok_per_sec'] == 1650000 and len(rows) == 12
    assert logs.first_losses(log) == {i: v for i, v in enumerate(source.R2['first_losses'])}
    assert logs.steady_stats(rows)['steps_parsed'] == 12 and logs.summary('no summary') == {}
