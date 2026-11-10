"""Node-side ENV-A programs (run on the leased WINH node under the env's Python, which has torch).

Never imported by driver-side code: `score_runner.py` imports torch and runs the task's own score.py, so importing
this package on the driver PC is avoided (the EnvARunner uploads and runs these files on the node instead)."""
