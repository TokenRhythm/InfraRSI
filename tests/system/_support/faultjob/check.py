"""The stand-in job's own test: fails when a file BREAK_TESTS exists in the tree, or when the rendered configuration
runs two data workers (a configuration the project's tests reject: the promotion must fall back to the next candidate)."""
import json
import os
import sys

bad = os.path.exists('BREAK_TESTS')
try:
    with open('config.json', encoding='utf-8') as f:
        bad = bad or json.load(f)['train'].get('workers') == 2
except (OSError, ValueError, KeyError):
    pass
sys.exit(1 if bad else 0)
