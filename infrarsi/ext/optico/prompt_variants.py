"""Request transforms for the B-NAT shim.

D-OC-15 `strip_think_format` (DECLARED, decision D7, issue #19; default of the B-NAT driver and of F-OPTICO):
claude-opus-5-5's API safeguard refuses OptiCo's give_sugg request verbatim ([reasoning_extraction]: the prompt
demands the model's step-by-step reasoning inside <think> tags; 9/9 refused in the G-B smoke and its diagnosis,
2026-10-21), while the same request without those output-format lines was accepted 4/4. The transform drops every
line that asks for reasoning inside think tags or holds the reasoning placeholder; nothing else changes. analyzer.py's
parsers are unaffected (`split('</think>')[-1]` keeps the whole answer when no tag is present; the 'The suggestions
are:' / 'The modified parameters are:' markers stay requested). The rule is content-agnostic (no OptiCo text is
stored here; OptiCo-public has no LICENSE).
"""
from __future__ import annotations

import re

_THINK_LINE = re.compile(r'</?think>|think tags|reasoning goes here', re.I)


def strip_think_format(text: str) -> tuple[str, int]:
    """(text without the lines demanding <think>-tagged reasoning, number of lines dropped)."""
    lines = text.split('\n')
    keep = [ln for ln in lines if not _THINK_LINE.search(ln)]
    return '\n'.join(keep), len(lines) - len(keep)


TRANSFORMS = {'strip_think_format': strip_think_format}
DEVIATION = {'strip_think_format': 'D-OC-15'}
