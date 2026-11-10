"""Agent harness: LLM backends, roles, tools, context, budgets and the optimize() search loop."""
from .budget import Budget, SearchLedger, VirtualClock
from .harness_runner import HarnessScreenRunner
from .judge import ReceiptJudge, RewardEngineJudge, Verdict, default_judge
from .optimize import optimize
from .runner import FakeScreenRunner, ScreenRunner, synthetic_receipt
from .search import SearchSession, SearchTrace

__all__ = ['Budget', 'FakeScreenRunner', 'HarnessScreenRunner', 'ReceiptJudge', 'RewardEngineJudge', 'ScreenRunner', 'SearchLedger', 'SearchSession',
           'SearchTrace', 'Verdict', 'VirtualClock', 'default_judge', 'optimize', 'synthetic_receipt']


def __getattr__(name):
    """DISPATCH_HANDLERS (RUN_MATRIX arm -> handler) for infrarsi.dispatch.handlers, built on first access."""
    if name == 'DISPATCH_HANDLERS':
        from .arms import agent_handlers as _build
        return _build()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
