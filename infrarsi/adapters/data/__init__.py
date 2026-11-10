"""Packed data for the UpdateBench v0 public tasks (INTERFACES §4b, design §1.2).

prepare(data_key, tokenizer_dir, out_root, seq_len, packs_total) -> manifest path
PackedTHDDataset(manifest_path, plan) -> pack(i), order(), update_assignment(...), update_token_counts(...)
update_assignment(update_index, packs_per_update, dp_rank, dp_size, micro_batch_size, order=None) -> pack ids
manifest_for_task(task) -> manifest path from INDEX.json (task id -> manifest), checked against the task
Batch build of every task + INDEX.json: python -m infrarsi.adapters.data.build_index --specs <task_specs dir>
"""
from .dataset import PackedTHDDataset, manifest_for_task, update_assignment, update_slots
from .packing import (IGNORE_INDEX, PACKER_VERSION, SCHEMA_PACKED, FirstFitPacker, output_name, permutation,
                      prepare, verify_dir)
from .sources import DATA_KEYS, TEMPLATES
from .tokenizer import HFTokenizer

__all__ = ['DATA_KEYS', 'TEMPLATES', 'IGNORE_INDEX', 'PACKER_VERSION', 'SCHEMA_PACKED', 'FirstFitPacker',
           'HFTokenizer', 'PackedTHDDataset', 'manifest_for_task', 'output_name', 'permutation', 'prepare',
           'update_assignment', 'update_slots', 'verify_dir']
