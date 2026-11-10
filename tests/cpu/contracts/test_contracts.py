from infrarsi.contracts import TaskSpec, RunRequest, load_all_tasks, digest, check_receipt_shape


def test_all_task_specs_load():
    tasks = load_all_tasks()
    assert len(tasks) == 18
    for tid, t in tasks.items():
        assert t.id == tid
        assert t.gpus >= 1 and t.seq_len >= 4096


def test_run_request_defaults_and_roundtrip():
    t = TaskSpec.load('D4')
    r = RunRequest(run_id='E0-CAL-D4-FR', purpose='screen', task=t.raw, config={'tensor_model_parallel_size': 2},
                   nodes=['WINH1'], gpus_per_node=8)
    assert r.steps == 30 and r.window == (17, 30) and r.world_size == 8
    d = r.to_dict()
    r2 = RunRequest.from_dict(d)
    assert r2.to_dict() == d
    p = RunRequest(run_id='x', purpose='probe', task=t.raw, config={}, nodes=['WINH1'], gpus_per_node=8)
    assert p.steps == 5 and p.window is None


def test_digest_stable_and_receipt_shape():
    assert digest({'b': 1, 'a': [1, 2]}) == digest({'a': [1, 2], 'b': 1})
    assert 'missing per_step' in check_receipt_shape({'schema': 'infrarsi-receipt-v1'})
