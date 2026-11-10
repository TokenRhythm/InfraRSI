"""Small JSON analysis language using Python syntax, never Python exec/eval (from rsi_optimizer_harness_v1).

Package tools are written in it: a script reads `data` (a receipt view) and `return`s a JSON value. It is a
limited common tool, not a general Python sandbox: no functions, imports, attributes or I/O.
Values: finite numbers, strings, bool, None, lists, dictionaries. Statements: assign, augmented assign, if, for,
and return.
"""
import ast
import json
import math
import operator


class DriverRejection(ValueError):
    pass

TOOL_SCHEMA = {
    'language': 'json-analysis-python-subset-v1',
    'input': 'JSON value named data',
    'output': 'JSON value returned at top level',
    'statements': ['assign', 'augmented assign', 'if', 'for', 'return'],
    'expressions': ['constant', 'name', 'list', 'dict', 'subscript', 'slice',
                    'arithmetic', 'comparison', 'boolean', 'conditional'],
    'functions': ['len', 'sum', 'min', 'max', 'abs', 'sorted', 'range', 'round'],
    'limits': {'source_bytes': 16000, 'json_bytes': 262144, 'operations': 50000,
               'range_length': 4096, 'integer_bits': 256, 'round_ndigits': [-32, 32]},
}


class AnalysisError(DriverRejection):
    pass


class _Return(Exception):
    def __init__(self, value):
        self.value = value


def checked_json(value):
    """All intermediate values are JSON sized, with bounded integer arithmetic."""
    def visit(v, depth=0):
        if depth > 32:
            raise AnalysisError('value depth limit')
        if type(v) is int and v.bit_length() > 256:
            raise AnalysisError('integer limit')
        if type(v) is float and not math.isfinite(v):
            raise AnalysisError('nonfinite value')
        if type(v) in (list, dict):
            if len(v) > 4096:
                raise AnalysisError('container length limit')
            for item in v.values() if type(v) is dict else v:
                visit(item, depth + 1)
            if type(v) is dict and any(type(k) is not str for k in v):
                raise AnalysisError('dictionary keys must be strings')
        elif type(v) not in (int, float, str, bool, type(None)):
            raise AnalysisError('non-JSON value')
    visit(value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(raw.encode('utf-8')) > TOOL_SCHEMA['limits']['json_bytes']:
        raise AnalysisError('JSON size limit')
    return value


def validate_source(source):
    if type(source) is not str or len(source.encode('utf-8')) > 16000:
        raise AnalysisError('source size/type')
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as exc:
        raise AnalysisError(str(exc)) from exc
    allowed = (ast.Module, ast.Assign, ast.AugAssign, ast.If, ast.For, ast.Return,
               ast.Expr, ast.Constant, ast.Name, ast.List, ast.Dict, ast.Subscript,
               ast.Slice, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
               ast.Call, ast.Load, ast.Store, ast.Add, ast.Sub, ast.Mult, ast.Div,
               ast.FloorDiv, ast.Mod, ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
               ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise AnalysisError('unsupported syntax: ' + type(node).__name__)
        if isinstance(node, ast.Name) and (node.id.startswith('_') or node.id in ('open', 'eval', 'exec')):
            raise AnalysisError('forbidden name')
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or
                                          node.func.id not in TOOL_SCHEMA['functions'] or node.keywords):
            raise AnalysisError('unsupported function')
    return tree


class Interpreter:
    def __init__(self, data):
        self.env = {'data': checked_json(json.loads(json.dumps(data, allow_nan=False)))}
        self.remaining = TOOL_SCHEMA['limits']['operations']

    def tick(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise AnalysisError('operation limit')

    def expr(self, node):
        self.tick()
        if isinstance(node, ast.Constant):
            result = node.value
        elif isinstance(node, ast.Name):
            result = self.env[node.id]
        elif isinstance(node, ast.List):
            result = [self.expr(v) for v in node.elts]
        elif isinstance(node, ast.Dict):
            result = {self.expr(k): self.expr(v) for k, v in zip(node.keys, node.values)}
        elif isinstance(node, ast.Subscript):
            obj = self.expr(node.value)
            if isinstance(node.slice, ast.Slice):
                s = node.slice
                key = slice(*(self.expr(x) if x else None for x in (s.lower, s.upper, s.step)))
            else:
                key = self.expr(node.slice)
            result = obj[key]
        elif isinstance(node, ast.BinOp):
            a, b = self.expr(node.left), self.expr(node.right)
            if type(a) not in (int, float) or type(b) not in (int, float):
                # Sequence multiplication is excluded before it could allocate.
                if not (isinstance(node.op, ast.Add) and type(a) is type(b) and type(a) in (str, list)):
                    raise AnalysisError('arithmetic requires numbers')
                if len(a) + len(b) > 4096:
                    raise AnalysisError('concatenation limit')
            ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                   ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
            result = ops[type(node.op)](a, b)
        elif isinstance(node, ast.UnaryOp):
            result = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}[type(node.op)](self.expr(node.operand))
        elif isinstance(node, ast.BoolOp):
            result = self.expr(node.values[0])
            for part in node.values[1:]:
                if isinstance(node.op, ast.And) and not result or isinstance(node.op, ast.Or) and result:
                    break
                result = self.expr(part)
        elif isinstance(node, ast.Compare):
            left, result = self.expr(node.left), True
            ops = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                   ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                   ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b}
            for op, right_node in zip(node.ops, node.comparators):
                right = self.expr(right_node)
                if not ops[type(op)](left, right):
                    result = False
                    break
                left = right
        elif isinstance(node, ast.IfExp):
            result = self.expr(node.body if self.expr(node.test) else node.orelse)
        elif isinstance(node, ast.Call):
            args = [self.expr(arg) for arg in node.args]
            name = node.func.id
            if name == 'round' and len(args) == 2:
                if type(args[1]) is not int or not -32 <= args[1] <= 32:
                    raise AnalysisError('round ndigits limit')
            if name == 'range':
                seq = range(*args)
                if len(seq) > 4096:
                    raise AnalysisError('range limit')
                result = list(seq)
            else:
                funcs = {'len': len, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
                         'sorted': sorted, 'round': round}
                result = funcs[name](*args)
        else:
            raise AnalysisError('unsupported expression')
        # Existing values/subvalues are already checked. Re-serializing the
        # complete input for each Name access makes a short loop quadratic.
        if isinstance(node, (ast.Name, ast.Subscript, ast.Compare, ast.BoolOp, ast.IfExp)):
            return result
        return checked_json(result)

    def assign(self, node, value):
        if not isinstance(node, ast.Name) or node.id == 'data' or node.id in TOOL_SCHEMA['functions']:
            raise AnalysisError('assign only new local names')
        self.env[node.id] = value

    def block(self, nodes):
        for node in nodes:
            self.tick()
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1:
                    raise AnalysisError('one assignment target')
                self.assign(node.targets[0], self.expr(node.value))
            elif isinstance(node, ast.AugAssign):
                value = self.expr(ast.BinOp(left=node.target, op=node.op, right=node.value))
                self.assign(node.target, value)
            elif isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test) else node.orelse)
            elif isinstance(node, ast.For):
                values = self.expr(node.iter)
                if type(values) is not list or node.orelse:
                    raise AnalysisError('for requires list, no else')
                for value in values:
                    self.assign(node.target, value)
                    self.block(node.body)
            elif isinstance(node, ast.Return):
                raise _Return(self.expr(node.value) if node.value else None)
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            else:
                raise AnalysisError('unsupported statement')


def run(source, data):
    try:
        interpreter = Interpreter(data)
        try:
            interpreter.block(validate_source(source).body)
        except _Return as result:
            return checked_json(result.value)
        raise AnalysisError('analysis must return JSON')
    except AnalysisError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, OverflowError, RecursionError) as exc:
        raise AnalysisError(str(exc)) from exc
