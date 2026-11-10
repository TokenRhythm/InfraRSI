"""The knob-only study's code-track gate (FinalistBase._can_code, trees pc_code_k*): a project whose open layers exclude
patches (code.mutable empty: the hooks' layers are ('L0',)) has no code track - no model calls held back for one, no patch
asked; a project with patches open, or a proposer without hooks, keeps llmopt's gate as before."""
from types import SimpleNamespace

from infrarsi.ext.common.session_hooks import layers_for
from infrarsi.search.finalist_base import FinalistBase


class _Proposer:
    def __init__(self, layers=None, driver=object()):
        self.driver = driver
        if layers is not None:
            self.hooks = SimpleNamespace(layers=layers)

    def hypotheses(self, *a, **k):          # (present: the proposer can make patch calls)
        return []


def can_code(proposer):
    inst = object.__new__(FinalistBase)     # (no __init__: the gate reads the proposer only)
    inst.proposer = proposer
    return inst._can_code()


def test_knob_only_project_has_no_code_track():
    assert layers_for('knobs') == ('L0',)
    assert can_code(_Proposer(layers=layers_for('knobs'))) is False


def test_patch_layers_keep_the_code_track():
    assert can_code(_Proposer(layers=layers_for('knobs+patch'))) is True
    assert can_code(_Proposer(layers=layers_for('patch'))) is True


def test_no_hooks_or_no_model():
    assert can_code(_Proposer()) is True                     # no hooks: llmopt's gate alone (both layers)
    assert can_code(_Proposer(layers=('L0', 'L1'), driver=None)) is False   # no model: no code track, as before


def test_knob_only_ladder_holds_no_allowance_for_proofs():
    """KNOB_r1 P2: with patches closed the ladder's code reserve is released (the whole low-fidelity allowance is the
    knob side's), and _can_code is off; with patches open, or no hooks, CODE_RESERVE stays as before."""
    from finalist_helpers import CodeProposer, Dummy
    from infrarsi.search.ladder import CODE_RESERVE

    knob = CodeProposer()
    knob.hooks = SimpleNamespace(layers=layers_for('knobs'))
    st = Dummy(knob)
    assert st.on['wcs_code'] and st.ladder.code_reserve == 0.0 and st._can_code() is False
    both = CodeProposer()
    both.hooks = SimpleNamespace(layers=layers_for('knobs+patch'))
    assert Dummy(both).ladder.code_reserve == CODE_RESERVE == 0.2 and Dummy(both)._can_code() is True
    assert Dummy(CodeProposer()).ladder.code_reserve == CODE_RESERVE      # (no hooks: the earlier behaviour)
