"""In-process frozen zone of the InfraRSI training process.

Timers, per-rank signals, the rank-0 receipt, invariants, gates, heartbeat, STOP handling and the
save/resume/export lifecycle live here. Agents may only replace files in ``runtime/overlays/`` (L1);
everything else is hashed into ``frozen_zone_digest``. Modules without a torch import stay importable
on the driver PC so their logic is covered by CPU tests.
"""
