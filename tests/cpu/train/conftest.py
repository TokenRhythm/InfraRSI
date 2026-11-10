"""Fixtures for the T-train CPU tests (no torch, no CUDA, no network)."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

import infrarsi.contracts as contracts
from infrarsi.contracts import RunRequest, TaskSpec, load_all_tasks

# On WINH the design tree is not next to the code; the dev upload ships task_specs separately.
if os.environ.get('INFRARSI_DESIGN_DIR'):
    contracts.DESIGN_DIR = Path(os.environ['INFRARSI_DESIGN_DIR'])


@pytest.fixture(scope='session')
def tasks():
    return load_all_tasks()


@pytest.fixture
def make_request(tasks):
    def _make(task_id, config, gpus=None, nodes=None, purpose='screen', **kw):
        task = copy.deepcopy(tasks[task_id].raw)
        nodes = nodes or ['WINH1']
        gpus = gpus or TaskSpec(task).gpus // len(nodes)
        return RunRequest(run_id=f'dev-T-train-{task_id}', purpose=purpose, task=task, config=config, nodes=nodes,
                          gpus_per_node=gpus, **kw)
    return _make
