"""Minimal JSON-schema validator (the subset our action schemas use; no third-party dependency)."""
from __future__ import annotations

import math
import re

_TYPES = {
    'object': lambda v: type(v) is dict,
    'array': lambda v: type(v) is list,
    'string': lambda v: type(v) is str,
    'integer': lambda v: type(v) is int,
    'number': lambda v: type(v) in (int, float) and math.isfinite(v),
    'boolean': lambda v: type(v) is bool,
    'null': lambda v: v is None,
}


def validate(value, schema: dict, path: str = '$') -> list[str]:
    errs: list[str] = []
    if not schema:
        return errs
    if 'oneOf' in schema:
        matches = [validate(value, s, path) for s in schema['oneOf']]
        ok = [m for m in matches if not m]
        if len(ok) != 1:
            # Report the branch selected by the 'action' discriminator if present, else a summary.
            if type(value) is dict and 'action' in value:
                for s, m in zip(schema['oneOf'], matches):
                    if s.get('properties', {}).get('action', {}).get('const') == value['action']:
                        return m or [f'{path}: matches several schemas']
                return [f'{path}.action: unknown action {value.get("action")!r}']
            return [f'{path}: matches {len(ok)} of oneOf (need exactly 1)']
        return errs
    if 'anyOf' in schema:
        if all(validate(value, s, path) for s in schema['anyOf']):
            return [f'{path}: matches none of anyOf']
    t = schema.get('type')
    if t is not None:
        types = t if type(t) is list else [t]
        if not any(_TYPES[x](value) for x in types):
            return [f'{path}: expected {"/".join(types)}, got {type(value).__name__}']
    if 'const' in schema and value != schema['const']:
        errs.append(f'{path}: expected {schema["const"]!r}')
    if 'enum' in schema and value not in schema['enum']:
        errs.append(f'{path}: {value!r} not in {schema["enum"]}')
    if type(value) is str:
        if 'minLength' in schema and len(value) < schema['minLength']:
            errs.append(f'{path}: shorter than {schema["minLength"]}')
        if 'maxLength' in schema and len(value) > schema['maxLength']:
            errs.append(f'{path}: longer than {schema["maxLength"]}')
        if 'pattern' in schema and not re.search(schema['pattern'], value):
            errs.append(f'{path}: does not match {schema["pattern"]}')
    if type(value) in (int, float) and type(value) is not bool:
        if 'minimum' in schema and value < schema['minimum']:
            errs.append(f'{path}: < {schema["minimum"]}')
        if 'maximum' in schema and value > schema['maximum']:
            errs.append(f'{path}: > {schema["maximum"]}')
    if type(value) is list:
        if 'minItems' in schema and len(value) < schema['minItems']:
            errs.append(f'{path}: fewer than {schema["minItems"]} items')
        if 'maxItems' in schema and len(value) > schema['maxItems']:
            errs.append(f'{path}: more than {schema["maxItems"]} items')
        if 'items' in schema:
            for i, item in enumerate(value):
                errs += validate(item, schema['items'], f'{path}[{i}]')
    if type(value) is dict:
        props = schema.get('properties', {})
        for k in schema.get('required', []):
            if k not in value:
                errs.append(f'{path}: missing {k}')
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], f'{path}.{k}')
            else:
                extra = schema.get('additionalProperties', True)
                if extra is False:
                    errs.append(f'{path}: unexpected key {k}')
                elif type(extra) is dict:
                    errs += validate(v, extra, f'{path}.{k}')
        if 'maxProperties' in schema and len(value) > schema['maxProperties']:
            errs.append(f'{path}: more than {schema["maxProperties"]} keys')
    return errs
