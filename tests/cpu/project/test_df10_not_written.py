"""R2-09 (2026-10-24): a run that cannot be written back as experience (the toy: no model.params_b / workload.seq_len)
said nothing - result.json had written: null and `exp export` answered "needs a run directory with a written pack".
The result, the report and export now name the missing facts."""
import pytest

from infrarsi.cli.main import main
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import markdown

from test_project import toy  # noqa: F401 - the toy project fixture


def test_a_run_without_model_facts_says_why_it_wrote_no_pack(toy, tmp_path):
    res = run_project(Project.load(toy.root), strategy='random', budget={'screens': 2}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    why = (res.get('experience') or {}).get('not_written')
    assert why and why.startswith('the manifest has no ') and 'workload.seq_len' in why
    assert f'This run was not written back as experience: {why}.' in markdown(res)
    with pytest.raises(SystemExit, match='wrote no experience pack: the manifest has no '):
        main(['exp', 'export', str(res['run_dir']), '--out', str(tmp_path / 'p.jsonl')])
