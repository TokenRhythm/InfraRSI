"""The verifier: speed is the only objective, `tolerance` the one human knob, and a generic correctness checklist
decides validity and feeds its failures back to the search (revamp design A)."""
from .checklist import (CHECKS, DEFAULT_REQUIRED, FAIL, INFO, OFF, PASS, PROMOTION_IDS, SCREEN_IDS, UNCHECKED, CheckDef,
                        blocking, coverage, feedback, result)
from .verifier import SCHEMA, Verifier, evidence, margin_of

__all__ = ['CHECKS', 'CheckDef', 'DEFAULT_REQUIRED', 'FAIL', 'INFO', 'OFF', 'PASS', 'PROMOTION_IDS', 'SCHEMA', 'SCREEN_IDS',
           'UNCHECKED', 'Verifier', 'blocking', 'coverage', 'evidence', 'feedback', 'margin_of', 'result']
