"""Codex R3 HIGH (builder X4): the dtype census exempts A_log / dt_bias / expert_bias from the bf16 parameter contract
only when exactly fp32, censuses them apart (model_fp32) with their gradients, and counts any other dtype as a
parameter so the census fails. runtime/gates.dtype_census imports torch lazily; a minimal stand-in module (dtypes,
is_tensor) lets the policy run without torch."""
import sys
from types import SimpleNamespace

import pytest

from infrarsi.runtime import gates


class DType:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f'torch.{self.name}'

    __repr__ = __str__


BF16, FP32, FP16, FP64 = DType('bfloat16'), DType('float32'), DType('float16'), DType('float64')


class T:
    def __init__(self, dtype, main_grad=None, requires_grad=True, device='cuda'):
        self.dtype, self.main_grad, self.requires_grad = dtype, main_grad, requires_grad
        self.device = SimpleNamespace(type=device)


class Model:
    def __init__(self, params, buffers=()):
        self.params, self.buffers = list(params), list(buffers)

    def named_parameters(self):
        return iter(self.params)

    def named_buffers(self):
        return iter(self.buffers)


@pytest.fixture
def fake_torch(monkeypatch):
    stub = SimpleNamespace(bfloat16=BF16, float32=FP32, float16=FP16, float64=FP64, is_tensor=lambda x: isinstance(x, T))
    monkeypatch.setitem(sys.modules, 'torch', stub)
    return stub


def _optimizer():
    inner = SimpleNamespace(state={0: {'exp_avg': T(FP32), 'exp_avg_sq': T(FP32)}})
    return SimpleNamespace(shard_fp32_from_float16_groups=[[T(FP32)]], optimizer=inner)


def _gdn(a_log=FP32, dt_bias=FP32, grad=FP32, buffers=()):
    return Model([('decoder.layers.0.mlp.linear_fc1.weight', T(BF16, T(FP32))),
                  ('decoder.layers.0.self_attention.A_log', T(a_log, T(grad))),
                  ('decoder.layers.0.self_attention.dt_bias', T(dt_bias, T(grad)))], buffers)


def test_exactly_fp32_gdn_terms_are_censused_apart_with_their_gradients(fake_torch):
    census = gates.dtype_census([_gdn()], _optimizer())
    assert gates.census_problems(census) == []
    assert census['counts']['model_fp32'] == {'torch.float32': 2} and census['counts']['param'] == {'torch.bfloat16': 1}
    assert census['counts']['grad'] == {'torch.float32': 3}                          # the exempt params' grads too
    assert census['model_defined_fp32_params'] == ['decoder.layers.0.self_attention.A_log:torch.float32',
                                                   'decoder.layers.0.self_attention.dt_bias:torch.float32']


@pytest.mark.parametrize('dtype', [FP16, FP64])
def test_any_other_dtype_is_not_an_exemption(fake_torch, dtype):
    census = gates.dtype_census([_gdn(a_log=dtype)], _optimizer())
    assert census['counts']['param'] == {'torch.bfloat16': 1, str(dtype): 1}
    assert any('param expected torch.bfloat16' in p for p in gates.census_problems(census))
    assert all('A_log' not in n for n in census['model_defined_fp32_params'])


def test_bf16_gdn_terms_are_ordinary_parameters(fake_torch):
    """Megatron-Core 0.18.2 builds A_log / dt_bias with params_dtype (bf16 here): no exemption needed or granted."""
    census = gates.dtype_census([_gdn(a_log=BF16, dt_bias=BF16)], _optimizer())
    assert gates.census_problems(census) == [] and census['counts']['param'] == {'torch.bfloat16': 3}
    assert census['counts']['model_fp32'] == {} and census['model_defined_fp32_params'] == []


def test_gradients_of_exempt_terms_are_checked(fake_torch):
    census = gates.dtype_census([_gdn(grad=FP16)], _optimizer())
    assert any('grad expected torch.float32' in p for p in gates.census_problems(census))


def test_router_expert_bias_buffer_must_be_fp32(fake_torch):
    ok = gates.dtype_census([_gdn(buffers=[('decoder.layers.0.mlp.router.expert_bias', T(FP32)),
                                           ('decoder.layers.0.mlp.router.local_tokens_per_expert', T(FP32))])], _optimizer())
    assert gates.census_problems(ok) == [] and ok['counts']['model_fp32'] == {'torch.float32': 3}
    bad = gates.dtype_census([_gdn(buffers=[('decoder.layers.0.mlp.router.expert_bias', T(BF16))])], _optimizer())
    assert any('model_fp32 expected torch.float32' in p for p in gates.census_problems(bad))


def test_dense_models_need_no_model_fp32_entries(fake_torch):
    dense = Model([('decoder.layers.0.mlp.linear_fc1.weight', T(BF16, T(FP32)))])
    census = gates.dtype_census([dense], _optimizer())
    assert gates.census_problems(census) == [] and census['counts']['model_fp32'] == {}
    assert 'model_fp32' in gates.OPTIONAL_CENSUS and gates.EXPECTED_DTYPES['model_fp32'] == 'torch.float32'
