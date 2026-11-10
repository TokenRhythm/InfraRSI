"""D2-R204c (2026-10-24): the Wan reference stopped at update 29 on an NVLink fault ("Invalid access of peer GPU
memory over nvlink or a hardware error") and the report's hint pointed at the configuration; the F phase's 1-GPU jobs
shared torchrun's port 29500 (EADDRINUSE). A failed run's hint names such a cause when the log's error line says so."""
from infrarsi.verify import Verifier
from infrarsi.verify.checklist import BY_ID


def _failed(error):
    return Verifier()._completes({'steps': 30, 'steps_completed': 29, 'exit_code': 1, 'events': {'error': error}})


def test_a_hardware_fault_or_a_taken_port_is_named():
    r = _failed('torch.AcceleratorError: CUDA error: Invalid access of peer GPU memory over nvlink or a hardware error')
    assert r['status'] == 'fail' and r['hint'].startswith('The GPU driver reported a hardware or interconnect fault')
    r = _failed('torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. '
                'port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use')
    assert 'its own (torchrun --master-port' in r['hint']
    assert _failed('torch.OutOfMemoryError: CUDA out of memory.')['hint'] == BY_ID['completes'].hint
    assert _failed(None)['hint'] == BY_ID['completes'].hint
