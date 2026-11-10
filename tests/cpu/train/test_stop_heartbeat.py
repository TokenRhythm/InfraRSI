import json
import time

import pytest

from infrarsi.runtime import stop
from infrarsi.runtime.heartbeat import Heartbeat
from infrarsi.runtime.timers import PhaseClock


def test_stop_flags(tmp_path):
    mon = stop.StopMonitor(tmp_path, wall_seconds_cap=3600)
    assert mon.flag() == stop.NONE and stop.read_stop(tmp_path) is None
    (tmp_path / 'STOP.json').write_text(json.dumps({'by': 'driver', 'reason': 'test'}))
    assert mon.flag() == stop.STOP_REQUESTED
    assert mon.flag(poll_file=False) == stop.NONE          # non-zero ranks rely on the all-gather
    assert stop.read_stop(tmp_path)['by'] == 'driver'
    (tmp_path / 'STOP.json').write_text('{not json')
    assert stop.read_stop(tmp_path) == {} and mon.flag() == stop.STOP_REQUESTED
    capped = stop.StopMonitor(tmp_path / 'x', wall_seconds_cap=10, started=time.monotonic() - 11)
    assert capped.flag() == stop.WALL_CAP and stop.StopMonitor.reason(stop.WALL_CAP).startswith('wall')


def test_heartbeat_schema(tmp_path):
    hb = Heartbeat(tmp_path, enabled=True, interval=0.05)
    hb.beat('build')
    d = json.loads((tmp_path / 'heartbeat.json').read_text())
    assert d['schema'] == 'infrarsi-heartbeat-v1' and d['phase'] == 'build' and d['step'] is None
    hb.start()
    hb.beat('train', 3)
    time.sleep(0.2)
    hb.close('done')
    d = json.loads((tmp_path / 'heartbeat.json').read_text())
    assert d['phase'] == 'done' and d['step'] == 3 and isinstance(d['pid'], int)
    with pytest.raises(ValueError):
        hb.beat('compiling')
    silent = Heartbeat(tmp_path / 'r1', enabled=False)
    silent.beat('train', 1)
    assert not (tmp_path / 'r1' / 'heartbeat.json').exists()


def test_phase_clock():
    clock = PhaseClock()
    with clock.phase('load'):
        time.sleep(0.01)
    clock.add('load', 1.0)
    snap = clock.snapshot()
    assert snap['load'] >= 1.01 and snap['total'] >= 0.01
