"""Trial D1-15 (Opus over the Claude CLI): a hypothesis call whose thinking hit the output cap was resumed by the CLI
itself (a synthetic user turn, "Output token limit hit. Resume directly ...") and answered in the second turn; the
one-turn check rejected the valid answer."""
import json

from infrarsi.agent.backends.claude_cli import parse_stream
from agent_helpers import FIXTURES

HAIKU = (FIXTURES / 'stream_haiku_5_5_real.jsonl').read_text(encoding='utf-8')
RESUME = {'type': 'user', 'message': {'role': 'user', 'content': [{'type': 'text', 'text': 'Output token limit hit. '
          'Resume directly - no apology, no recap of what you were doing.'}]}, 'parent_tool_use_id': None,
          'session_id': 's', 'isSynthetic': True}


def _with_second_turn(user_event):
    ev = [json.loads(x) for x in HAIKU.splitlines() if x.strip()]
    i = next(k for k, e in enumerate(ev) if e['type'] == 'assistant')
    ev.insert(i + 1, user_event)
    ev[-1]['num_turns'] = 2
    return '\n'.join(json.dumps(e) for e in ev) + '\n'


def test_the_clis_own_resume_turn_is_accepted():
    action, f = parse_stream(_with_second_turn(RESUME), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert f['validation_error'] is None and action['action'] == 'propose_config' and f['resumed_turns'] == 1


def test_any_other_second_turn_is_still_rejected():
    other = {**RESUME, 'isSynthetic': False}
    action, f = parse_stream(_with_second_turn(other), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action is None and f['validation_error'] == 'unexpected turns or permission denials'
    tool = {**RESUME, 'message': {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't', 'content': 'x'}]}}
    action, f = parse_stream(_with_second_turn(tool), model='claude-haiku-5-5', exit_code=0, status='RETURNED')
    assert action is None and f['resumed_turns'] == 0
