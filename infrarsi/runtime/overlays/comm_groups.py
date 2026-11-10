"""Communicator hooks (L1). Default: Megatron-Core process groups and DDP buckets as configured by L0."""


def parallel_init_kwargs(ctx) -> dict:
    """Extra kwargs for parallel_state.initialize_model_parallel (e.g. nccl_communicator_config_path)."""
    return {}


def ddp_config_overrides(ctx) -> dict:
    """DistributedDataParallelConfig bucket/overlap fields (keys limited by overlays.ALLOWED)."""
    return {}
