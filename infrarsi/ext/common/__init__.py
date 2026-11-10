"""Shared layer of the external environments (EXT_INTERFACES §2-§6): task contract, runner base, judge, guards,
trajectory scoring, remote execution, the native-method basic agent and the SessionHooks seam."""
from .env_judge import SCHEMA_RECEIPT, EnvJudge, metric_value
from .env_runner import EnvRun, EnvRunner, candidate_digest
from .env_task import EnvTask, check_spec
from .guards import GuardResult, combine
from .remote_exec import GpuLease, LocalNodeOps, NodeOps, WinhNodeOps, nobody_cmd
from .session_hooks import SessionHooks

__all__ = ['EnvJudge', 'EnvRun', 'EnvRunner', 'EnvTask', 'GpuLease', 'GuardResult', 'LocalNodeOps', 'NodeOps',
           'SCHEMA_RECEIPT', 'SessionHooks', 'WinhNodeOps', 'candidate_digest', 'check_spec', 'combine', 'metric_value',
           'nobody_cmd']
