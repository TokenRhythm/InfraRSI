"""D2-R201b (2026-10-24): a screening design that did not run is named in the report (DOE's `why`)."""
from infrarsi.project import Project
from infrarsi.project.flow import run_project
from infrarsi.project.report import markdown

from test_project import toy  # noqa: F401 - the toy project fixture


def test_the_report_says_why_no_screening_design_ran(toy, tmp_path):
    res = run_project(Project.load(toy.root), strategy='doe', budget={'screens': 2}, out_root=tmp_path / 'out',
                      log=lambda m: None)
    why = (res['search'].get('design') or {}).get('why')
    assert why and f'Screening design: not run: {why}.' in markdown(res)
