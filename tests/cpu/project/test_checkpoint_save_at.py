"""checkpoint.save_at: one rule everywhere (the manifest's value, else half the project's steps), and a manifest check
that keeps it inside the run (found while init learned to draft checkpoint blocks, 2026-10-26)."""
import shutil
from pathlib import Path

import pytest
import yaml

from infrarsi.project import Project, ProjectRunner
from infrarsi.project.flow import make_verifier
from infrarsi.project.manifest import ManifestError

TOY = Path(__file__).resolve().parents[3] / 'examples' / 'toy_cpu'


def toy(tmp_path, **checkpoint):
    """A copy of the toy whose checkpoint block is replaced by `checkpoint` (its save and resume templates kept)."""
    root = tmp_path / 'toy'
    shutil.copytree(TOY, root)
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    doc['checkpoint'] = {'save': doc['checkpoint']['save'], 'resume': doc['checkpoint']['resume'], **checkpoint}
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    return root


def test_the_default_save_at_is_half_the_project_steps_also_for_the_save_run(tmp_path):
    """Without checkpoint.save_at the save run (which runs save_at updates) used to render {save_at} as half its own
    length: a checkpoint a quarter in, while the resume check expected one half-way; the resume then failed."""
    p = Project.load(toy(tmp_path))                      # the toy has 24 updates: a checkpoint after update 12
    assert p.save_at == 12
    runner = ProjectRunner(p, out_dir=tmp_path / 'out', log=lambda m: None)
    cmd = runner.command(p.reference_config(), cfg_rel='config.yaml', steps=p.save_at, run_dir=str(tmp_path / 'r'),
                         extra_args=p.raw['checkpoint']['save'], ckpt_dir=str(tmp_path / 'ck'))
    assert '--save-at 12' in cmd and '--steps 12' in cmd
    assert Project.load(toy(tmp_path / 'b', save_at=5)).save_at == 5


def test_a_given_save_at_must_be_an_int_inside_the_run(tmp_path):
    """A null, a bool, a float and a numeric string (a '12' used to render like 12) are refused with the same line as a
    value outside 1..steps-1; only an absent key takes the default."""
    for bad in (0, 24, 30, 'x', '12', True, 2.5, None):
        with pytest.raises(ManifestError, match='checkpoint.save_at must be an int from 1 to workload.steps - 1'):
            Project.load(toy(tmp_path / str(bad), save_at=bad))
    assert Project.load(toy(tmp_path / 'ok', save_at=23)).save_at == 23


def test_no_checkpoint_block_means_no_save_at(tmp_path):
    root = toy(tmp_path)
    doc = yaml.safe_load((root / 'infrarsi.yaml').read_text(encoding='utf-8'))
    del doc['checkpoint']
    (root / 'infrarsi.yaml').write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
    p = Project.load(root)
    assert p.save_at is None
    runner = ProjectRunner(p, out_dir=tmp_path / 'out', log=lambda m: None)   # a {save_at} in a command still renders
    assert '--save-at 12' in runner.command(p.reference_config(), cfg_rel='config.yaml', steps=24, run_dir='r',
                                             extra_args='--save-at {save_at}')
    assert '--save-at 5' in runner.command(p.reference_config(), cfg_rel='config.yaml', steps=11, run_dir='r',
                                            extra_args='--save-at {save_at}')    # half of this run, as before


def test_a_promotion_resumes_from_the_default_checkpoint(tmp_path, monkeypatch):
    """The toy for real with checkpoint.save_at omitted: the save run stops and saves after update 12 and the resumed
    run continues at 13, so resume_continues passes. Before the fix the save run saved after update 6 and the resumed
    run started at 7 while the check expected the checkpoint after update 12: every such promotion failed."""
    monkeypatch.setenv('TOY_TIME_SCALE', '0.004')
    p = Project.load(toy(tmp_path))
    t, ck, cfg = p.task(), p.raw['checkpoint'], p.reference_config()
    r = ProjectRunner(p, out_dir=tmp_path / 'runs', log=lambda m: None)
    full = r.run(t, cfg, None, 'rescore', 'r', candidate_id='c1')
    shared = r.shared_dir('r', 'ckpt-c1')
    r.run(t, cfg, None, 'save', 'r', candidate_id='c1', steps=p.save_at, extra_args=ck['save'], ckpt_dir=shared)
    resumed = r.run(t, cfg, None, 'resume', 'r', candidate_id='c1', extra_args=ck['resume'], ckpt_dir=shared)
    items = make_verifier(p).promotion(full, full, resume=resumed, save_at=p.save_at)
    item = next(i for i in items if i['id'] == 'resume_continues')
    assert item['status'] == 'pass' and 'over updates 13..24' in item['detail'], item
