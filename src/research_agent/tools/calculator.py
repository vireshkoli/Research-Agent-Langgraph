"""Arithmetic via a whitelisted AST walk. No exec, no eval, no subprocess.

This exists so that `code_execution` is not on the critical path for the thing agents
actually need numbers for. Almost every arithmetic step in a research task is a single
expression, and answering it here costs microseconds and carries no sandbox risk at
all. `code_execution` remains available for genuine multi-step logic.

`eval()` on model output would be a remote code execution hole: `__import__('os')`
is a valid expression. This walks the parsed tree and rejects any node type not on
the allowlist, so there is no path to a name lookup, attribute access, or call
outside `FUNCTIONS`.
"""

import ast
import math
import operator
from typing import Any

from research_agent.tools.base import ToolResult, ToolSpec

# Exponentiation is the one operator here that can hang the process: 9**9**9 is a
# trivially short expression that allocates until the machine dies. Everything else
# is bounded by the size of its operands.
MAX_EXPONENT = 1000
MAX_EXPR_CHARS = 500

BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
}

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


class CalculatorError(Exception):
    """Raised for anything the whitelist rejects. Always surfaced as a ToolResult."""


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalculatorError(f"only numbers are allowed, got {type(node.value).__name__}")
        return node.value

    if isinstance(node, ast.BinOp):
        op = BINARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError(f"exponent {right} exceeds the limit of {MAX_EXPONENT}")
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = UNARY_OPS.get(type(node.op))
        if op is None:
            raise CalculatorError(f"unary {type(node.op).__name__} is not allowed")
        return op(_evaluate(node.operand))

    if isinstance(node, ast.Call):
        # Only bare names may be called. `node.func` being an Attribute is what
        # `(1).__class__` style escapes rely on, so it never reaches a lookup.
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only direct calls to allowed functions are permitted")
        if node.func.id not in FUNCTIONS:
            raise CalculatorError(f"unknown function {node.func.id!r}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")
        return FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])

    if isinstance(node, ast.Name):
        if node.id not in CONSTANTS:
            raise CalculatorError(f"unknown name {node.id!r}")
        return CONSTANTS[node.id]

    if isinstance(node, ast.Tuple | ast.List):
        return [_evaluate(element) for element in node.elts]

    raise CalculatorError(f"{type(node).__name__} is not allowed in an expression")


def calculate(expression: str) -> ToolResult:
    """Evaluate one arithmetic expression and return the result as text."""
    expression = (expression or "").strip()
    if not expression:
        return ToolResult.failure("empty expression")
    if len(expression) > MAX_EXPR_CHARS:
        return ToolResult.failure(f"expression exceeds {MAX_EXPR_CHARS} characters")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        return ToolResult.failure(f"could not parse {expression!r}: {exc.msg}")

    try:
        value = _evaluate(tree)
    except CalculatorError as exc:
        return ToolResult.failure(str(exc))
    except ZeroDivisionError:
        return ToolResult.failure("division by zero")
    except (OverflowError, ValueError, TypeError) as exc:
        return ToolResult.failure(f"{type(exc).__name__}: {exc}")

    return ToolResult(ok=True, content=f"{expression} = {value}", meta={"value": value})


SPEC = ToolSpec(
    name="calculator",
    description=(
        "Evaluate a single arithmetic expression and return the exact result. "
        "Supports + - * / // % **, parentheses, the constants pi/e/tau, and the "
        "functions abs round min max sum pow sqrt log log2 log10 exp sin cos tan "
        "floor ceil factorial. Prefer this over code_execution for arithmetic."
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The expression to evaluate, e.g. '405 - 123' or 'sqrt(2) * 100'.",
            }
        },
        "required": ["expression"],
        "additionalProperties": False,
    },
    fn=calculate,
    max_calls=10,
)
