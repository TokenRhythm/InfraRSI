"""discover: HardwareProfile (read-only SSH probe) and workload facts from HF configs."""
from .hardware import default_profile, hardware_profile, parse_node, summarize
from .workload import count_params, facts_from_config, model_dir, tp_candidates, workload_facts

__all__ = ['hardware_profile', 'default_profile', 'parse_node', 'summarize', 'workload_facts', 'facts_from_config',
           'count_params', 'tp_candidates', 'model_dir']
