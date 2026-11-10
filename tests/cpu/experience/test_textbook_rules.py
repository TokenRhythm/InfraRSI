"""The textbook pack (design P1-2, 2026-10-24): two rules the proposer reads - FP8 GEMM shapes (any job) and torchao
float8's dependence on compile (not Megatron's Transformer Engine)."""
from expjob import KNOBS, REF, megatron_log, project
from infrarsi.experience.pack import validate
from infrarsi.experience.store import SEED, Store, fingerprint, plan, read_lines


def test_the_pack_is_clean_and_reaches_the_payload(tmp_path):
    assert validate(read_lines(SEED / 'textbook-2026-10.jsonl')) == ([], [])
    store = Store.load(include=['textbook-2026-10'])
    mcore = plan(store, fingerprint(project(), {'log': str(megatron_log(tmp_path / 'train.log'))}), KNOBS, REF)
    assert [r['id'] for r in mcore['payload']['rules']] == ['textbook-2026-10/fp8-gemm-shape']
    fsdp = plan(store, fingerprint(project(framework='fsdp')), {}, {})
    assert {r['id'] for r in fsdp['payload']['rules']} == {'textbook-2026-10/fp8-gemm-shape',
                                                          'textbook-2026-10/fp8-torchao-compile'}
