"""S2 - the shipped experience: every seed pack validates, every mechanism it names is in the registry, no private
strings, the registries load and are consistent, `exp stats` runs."""
import json
import os
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from _support import cli
from infrarsi.experience import pack as P
from infrarsi.experience import registry
from infrarsi.experience.store import Store

SEED = Path(__file__).resolve().parents[2] / 'infrarsi' / 'experience' / 'seed'
DATA = SEED.parent / 'data'
PACKS = sorted(SEED.glob('*.jsonl'))


@pytest.mark.parametrize('pack', PACKS, ids=lambda p: p.stem)
def test_seed_pack_validates(pack):
    errs, warns = P.validate(P.read_lines(pack))
    assert not errs, errs[:10]
    private = [w for w in warns if 'private' in w]
    assert not private, private


def test_cli_validate_on_every_pack_at_once():
    res = cli.run('exp', 'validate', *PACKS)
    assert res.code == 0, res.text
    assert res.out.count('0 errors') == len(PACKS)


def test_every_mechanism_named_by_the_packs_is_registered():
    mechs = registry.mechanisms()
    ids = set(mechs) if isinstance(mechs, dict) else {m['id'] for m in mechs}
    unknown = Counter()
    for pack in PACKS:
        for line in P.read_lines(pack):
            for field in ('change', 'config'):
                for k in (line.get(field) or {}):
                    if k not in ids and not k.startswith('x-'):
                        unknown[(pack.stem, k)] += 1
    assert not unknown, dict(unknown)


def test_registries_load_and_are_consistent():
    mech = yaml.safe_load((DATA / 'mechanisms.yaml').read_text(encoding='utf-8'))
    items = mech['mechanisms'] if isinstance(mech, dict) and 'mechanisms' in mech else mech
    entries = list(items.items()) if isinstance(items, dict) else [(m['id'], m) for m in items]
    assert len(entries) >= 60
    ids = [k for k, _ in entries]
    assert len(ids) == len(set(ids)), 'duplicate mechanism ids'
    claims = Counter()
    for k, m in entries:
        assert P.MECH_ID.match(k), k
        assert (m or {}).get('class') in ('kernel', 'balance'), (k, (m or {}).get('class'))
        for fw, items in ((m or {}).get('aliases') or {}).items():
            assert fw in P.FRAMEWORKS or fw == 'any', (k, fw)
            for a in items:
                kind = next((t for t in ('flag', 'env', 'config') if t in a), None)
                assert kind, (k, fw, a)
                claims[(fw, kind, str(a[kind]))] += 1
    dup = {c: n for c, n in claims.items() if n > 1}
    print(f'\n{len(entries)} mechanisms; knob names claimed by more than one mechanism in one framework: {dup or "none"}')
    gpus = yaml.safe_load((DATA / 'gpus.yaml').read_text(encoding='utf-8'))
    assert gpus
    schema = json.loads((DATA / 'schema.json').read_text(encoding='utf-8'))
    assert schema
    assert registry.gpu_class('NVIDIA H100 80GB HBM3') and registry.gpu_class('NVIDIA H20')


def test_no_private_strings_or_internal_hosts_in_seeds():
    """The validator's private-string rule (addresses, home directories, tokens), plus any site-specific names given in
    INFRARSI_PRIVATE_NAMES (comma-separated, e.g. host names and accounts of the machines the packs were measured on)."""
    names = [n.strip() for n in os.environ.get('INFRARSI_PRIVATE_NAMES', '').split(',') if n.strip()]
    hosts = re.compile(r'\b(?:' + '|'.join(re.escape(n) for n in names) + r')\b', re.I) if names else None
    bad = []
    for pack in PACKS:
        text = pack.read_text(encoding='utf-8')
        for m in P.PRIVATE.finditer(text):
            bad.append((pack.stem, m.group(0)))
        for m in (hosts.finditer(text) if hosts else ()):
            bad.append((pack.stem, m.group(0)))
    assert not bad, bad[:10]


def test_store_loads_every_pack_without_skips_and_stats_runs():
    st = Store.load()
    assert len(st.packs) >= len(PACKS) and not st.skipped, st.skipped
    assert st.observations and st.effects and st.rules
    res = cli.run('exp', 'stats')
    assert res.code == 0 and f'{len(st.packs)} packs' in res.out and 'readable effects per mechanism' in res.out
