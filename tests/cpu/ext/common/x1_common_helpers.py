"""Shared fixtures of the ext CPU tests (X1): a toy external environment with knobs and file patches."""
import hashlib
import math

from infrarsi.agent.budget import VirtualClock
from infrarsi.ext.common.env_runner import EnvRun, EnvRunner
from infrarsi.ext.common.env_task import EnvTask
from infrarsi.ext.common.guards import GuardResult
from infrarsi.ext.common.remote_exec import LocalNodeOps

SHA = hashlib.sha256(b'frozen').hexdigest()
BASE_FILES = {'solution/run.py': 'import time\n\n\ndef main():\n    time.sleep(1.0)\n    return 42\n',
              'solution/config.yaml': 'workers: 1\nbatch: 8\n'}
KNOBS = {'type': 'object', 'additionalProperties': False,
         'properties': {'workers': {'type': 'integer', 'enum': [1, 2, 4, 8], 'default': 1},
                        'flash': {'type': 'boolean', 'enum': [False, True], 'default': False}}}


def toy_spec(**over) -> dict:
    spec = {'id': 'ENV-T', 'env': 'toyenv', 'split': 'external', 'arm': 'T-IS', 'metric_name': 'exec_time_s',
            'action_space': 'knobs+patch', 'knob_schema': KNOBS, 'frozen_files': {'score.py': SHA},
            'mutable_globs': ['solution/*'], 'max_patch_lines': 20,
            'budget': {'screens': 6, 'llm_tokens': 50_000, 'wall_seconds': 36000.0, 'gpu_hours': 10.0},
            'margin': {'kind': 'fixed', 'value': 0.05}, 'tau': math.log(1.5), 'world_size': 4,
            'facts': {'model': {'name': 'toy'}, 'hardware': {'gpu_type': 'H100', 'nodes': 1}, 'data': {'name': 'toy'}}}
    spec.update(over)
    return spec


def toy_task(**over) -> EnvTask:
    return EnvTask.from_dict(toy_spec(**over))


class ToyRunner(EnvRunner):
    """Synthetic environment: time = 100 / workers (x0.8 with flash), patches that write 'fast' are 2x faster,
    'cheat' trips a guard, 'crash' makes the judge invalid. The clock advances by the run time."""
    env = 'toyenv'
    deviations = ('D-T-1',)

    def __init__(self, ops=None, *, clock=None, **kw):
        self.vclock = clock or VirtualClock()
        super().__init__(ops or LocalNodeOps('.'), gpus=(0, 1, 2, 3), clock=self.vclock, **kw)
        self.seen = []

    def execute(self, task, config, overlay, purpose, run_id, attempt_id, candidate_id, **kw):
        self.seen.append({'config': config, 'overlay': overlay, 'candidate_id': candidate_id, 'purpose': purpose})
        t = 100.0 / config.get('workers', 1) * (0.8 if config.get('flash') else 1.0)
        text = ''.join(overlay.values())
        if 'fast' in text:
            t /= 2
        self.vclock.advance(t + 20)
        g = GuardResult()
        if 'cheat' in text:
            g.reject('G2', 'score.py changed')
        g.report('G5', checkpoint='ok')
        ok = 'crash' not in text
        return EnvRun(candidate_id=candidate_id, judge={'score': math.log(t), 'execution_time': t if ok else None},
                      judge_valid=ok, metric_value=t if ok else None, metric_raw={'score': math.log(t)},
                      judge_reasons=[] if ok else ['solution crashed'], guards=g, phases={'run': t},
                      extra={'summary': {'t': t}})


def local_ops(tmp_path, apps=None, gpu_mem=None):
    def probe():
        return {'hostname': 'fake', 'utc': 'x', 'error': None, 'apps': list(apps or []),
                'gpus': [{'index': i, 'uuid': f'G{i}', 'mem_used_mib': (gpu_mem or {}).get(i, 0)} for i in range(8)]}
    return LocalNodeOps(tmp_path, probe_fn=probe)
