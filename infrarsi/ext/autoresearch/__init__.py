"""ENV-C adapter: autoresearch (karpathy, MIT; 1 H100, 300 s fixed budget, val_bpb), design §4.

source (base tree, R2 facts) -> task (EnvTask per arm) -> hooks (InfraRSI task view, knob and patch guards)
-> runner (candidate directory, training as nobody, trusted judge) -> arms / native (C-IS, C-RND, C-NAT-*).
infra_only_check, knobs, logs and judge_files are pure (no torch); node/ holds the WINH-side scripts.
"""
