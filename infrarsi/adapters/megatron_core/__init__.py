"""Generic Megatron-Core training process for UpdateBench tasks.

Entry: ``torchrun ... -m infrarsi.adapters.megatron_core.generic_train --request <request.json>``.
The model comes from the HF checkpoint through the vendored ModelScope mcore-bridge; parallel layout and
performance knobs come from the request's L0 config, the objective from the frozen task contract.
"""
