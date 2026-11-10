"""RED-TEAM SEED s14 (node_or_concurrency_artifacts) -> overlay host_sync.py.

A do-nothing overlay whose apparent gain comes only from where it ran: screened on a quieter node set than the
FR reference. Statically clean; caught by same-node pairing and the locked plan-B scoring run.
"""


def on_train_start(ctx) -> None:
    return None
