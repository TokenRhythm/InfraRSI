import argparse
import json

from infrarsi.adapters.data.build_index import build, raw_problems, tokenizer_sources
from infrarsi.contracts import TaskSpec


def _model(root, name, done, tok=b'{"tok": 1}', cfg=b'{"eos_token": "e"}'):
    d = root / name
    d.mkdir()
    if tok is not None:
        (d / 'tokenizer.json').write_bytes(tok)
        (d / 'tokenizer_config.json').write_bytes(cfg)
    if done:
        (d / 'DOWNLOAD_DONE.json').write_text('{}')


def test_tokenizer_sources(tmp_path):
    _model(tmp_path, 'Qwen__A', True)
    _model(tmp_path, 'Qwen__B', False)                                  # downloading, identical tokenizer files
    _model(tmp_path, 'Qwen__C', False, cfg=b'{"eos_token": "x"}')      # downloading, different config
    _model(tmp_path, 'Qwen__D', False, tok=None)                       # nothing yet
    out = tokenizer_sources(tmp_path, ['Qwen/A', 'Qwen/B', 'Qwen/C', 'Qwen/D', 'Qwen/E'])
    assert out['Qwen/A'] == (tmp_path / 'Qwen__A', 'download complete')
    assert out['Qwen/B'][0] == tmp_path / 'Qwen__A' and 'identical' in out['Qwen/B'][1]
    assert all(out[r][0] is None for r in ('Qwen/C', 'Qwen/D', 'Qwen/E'))


def test_raw_problems(tmp_path):
    out = raw_problems(tmp_path, ['D-SFT-8K'])
    assert out['D-SFT-8K'] and 'missing' in out['D-SFT-8K']


def test_first_pass_with_everything_pending_writes_index(tmp_path):
    spec = TaskSpec(raw={'id': 'S2', 'model': {'repo_id': 'Qwen/Qwen3-4B'},
                         'data': {'key': 'D-SFT-8K', 'seq_len': 8192},
                         'workload': {'seq_len': 8192, 'packs_per_update': 8, 'packs_total': 384}})
    a = argparse.Namespace(raw_root=str(tmp_path / 'raw'), models_root=str(tmp_path / 'models'),
                           out_root=str(tmp_path / 'not' / 'yet' / 'packed'), command='cmd', force=False)
    pending = build({'S2': spec}, a)
    index = json.loads((tmp_path / 'not' / 'yet' / 'packed' / 'INDEX.json').read_text(encoding='utf-8'))
    assert set(pending) == set(index['pending']) == {'S2'} and index['tasks'] == {}
    assert index['pending']['S2']['command'] == 'cmd' and index['pending']['S2']['packs_total'] == 384
