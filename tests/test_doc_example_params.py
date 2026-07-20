"""Conformance guard: MCP tool-call examples in docs use only real params.

Sibling to ``test_skill_tool_conformance.py`` (which checks tool *names*).
This guard checks tool-call *parameters*: every ``arguments`` key in an MCP
tool-call example embedded in ``README.md`` or ``docs/*.md`` must be a real
parameter of that tool per the ``@mcp.tool()`` function signature in
``src/althing/mcp/server.py``.

Motivation: shipped examples drifted to invented parameters (``personas_pack``
instead of ``pack_id``; a ``stimulus`` field no tool accepts). A naive user who
copy-pasted them hit ``{"error": "No personas provided..."}`` or silently polled
the wrong personas. Tool-name conformance did not catch this because the *tool*
existed — only a keyword inside ``arguments`` was wrong. This guard closes that
gap and would have failed on all three drift cases.

Two example conventions are recognised (the docs use both):

* **Convention A** — ``{"tool": "<name>", "arguments": { ... }}``. The keys of
  ``arguments`` are validated against ``<name>``.
* **Convention B** — a bare arguments object preceded by a ``// <tool>`` comment
  (e.g. ``// MCP run_panel``). The object's top-level keys are validated against
  the tool named in the comment.

The accepted-parameter map is parsed from ``server.py`` via ``ast`` (never
hardcoded), so renaming or adding a tool parameter keeps this guard current.
Blocks that intentionally show error/response envelopes are skipped, and any
object that is not valid JSON (illustrative ``...`` elisions, response bodies)
is skipped rather than failing the guard.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = REPO_ROOT / "src" / "althing" / "mcp" / "server.py"

# Scope mirrors the task and the sibling guard: the README plus the top-level
# docs pages (non-recursive, matching test_skill_tool_conformance.py's glob).
_DOC_FILES: list[Path] = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]

# Fenced ```json / ```jsonc blocks. Non-greedy body capture up to the closing
# fence at the start of a line.
_FENCE_RE = re.compile(r"^```(?:jsonc|json)[^\n]*\n(.*?)^```", re.DOTALL | re.MULTILINE)

# A comment that describes a response/error/output is never a tool *call*.
_RESPONSE_MARKER_RE = re.compile(r"\b(response|responses|returns?|output|envelope|error)\b", re.IGNORECASE)

# Params present on the tool signature but not caller-supplied arguments.
_NON_ARG_PARAMS = frozenset({"ctx", "self"})


def _accepted_params_by_tool() -> dict[str, set[str]]:
    """Map each ``@mcp.tool()`` function name to its accepted parameter names.

    Parsed from the AST of server.py (no import, no ``mcp`` dependency at test
    time) so the accepted set is exactly the tool's real signature.
    """
    tree = ast.parse(SERVER_PY.read_text(encoding="utf-8"))
    tools: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if not _has_mcp_tool_decorator(node):
            continue
        args = node.args
        names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
        tools[node.name] = names - _NON_ARG_PARAMS
    assert len(tools) >= 10, (
        f"parsed only {len(tools)} @mcp.tool() signatures from {SERVER_PY} ({sorted(tools)}) — "
        "the AST walk is likely stale."
    )
    return tools


def _has_mcp_tool_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True when *node* carries an ``@mcp.tool(...)`` decorator."""
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if (
            isinstance(target, ast.Attribute)
            and target.attr == "tool"
            and isinstance(target.value, ast.Name)
            and target.value.id == "mcp"
        ):
            return True
    return False


ACCEPTED_PARAMS = _accepted_params_by_tool()
REAL_TOOLS = frozenset(ACCEPTED_PARAMS)


def _iter_json_blocks(text: str) -> list[str]:
    """Return the bodies of every fenced ```json / ```jsonc block in *text*."""
    return [m.group(1) for m in _FENCE_RE.finditer(text)]


def _extract_objects(block: str) -> list[tuple[str, str]]:
    """Yield ``(comment_free_object_json, preceding_comment)`` per top-level object.

    A single-pass scanner tracks string/comment/brace state so it survives
    nested objects, ``//`` and ``/* */`` comments (including URLs inside string
    values, which a naive comment strip would corrupt), and multiple objects in
    one fenced block. Comments seen at brace-depth 0 between objects are the
    "preceding comment" of the next object (Convention B's tool hint).
    """
    results: list[tuple[str, str]] = []
    clean: list[str] = []
    depth = 0
    in_str = False
    esc = False
    obj_start: int | None = None
    pending: list[str] = []
    comment_for_obj = ""
    i = 0
    n = len(block)
    while i < n:
        c = block[i]
        if in_str:
            clean.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == "/" and i + 1 < n and block[i + 1] == "/":
            j = block.find("\n", i + 2)
            if j == -1:
                j = n
            if depth == 0:
                pending.append(block[i + 2 : j])
            i = j
            continue
        if c == "/" and i + 1 < n and block[i + 1] == "*":
            j = block.find("*/", i + 2)
            end = n if j == -1 else j
            if depth == 0:
                pending.append(block[i + 2 : end])
            i = (end + 2) if j != -1 else n
            continue
        if c == '"':
            in_str = True
            clean.append(c)
            i += 1
            continue
        if c == "{":
            if depth == 0:
                obj_start = len(clean)
                comment_for_obj = " ".join(pending)
                pending = []
            depth += 1
            clean.append(c)
            i += 1
            continue
        if c == "}":
            if depth > 0:
                depth -= 1
                clean.append(c)
                if depth == 0 and obj_start is not None:
                    results.append(("".join(clean[obj_start:]), comment_for_obj))
                    obj_start = None
            i += 1
            continue
        clean.append(c)
        i += 1
    return results


def _identify_call(obj: object, comment: str) -> tuple[str, set[str]] | None:
    """Return ``(tool, arg_keys_to_check)`` for a recognised tool call, else None.

    Objects that carry an ``error`` / ``error_code`` marker (intentional error
    examples) are skipped, as are objects whose only reasonable identification
    would come from a response/output comment.
    """
    if not isinstance(obj, dict):
        return None
    if "error" in obj or "error_code" in obj:
        return None

    # Convention A: {"tool": "<name>", "arguments": {...}}
    tool = obj.get("tool")
    arguments = obj.get("arguments")
    if isinstance(tool, str) and tool in REAL_TOOLS and isinstance(arguments, dict):
        return tool, set(arguments.keys())

    # Convention B: bare object preceded by a `// <tool>` comment.
    if comment and not _RESPONSE_MARKER_RE.search(comment):
        named = [t for t in REAL_TOOLS if re.search(rf"\b{re.escape(t)}\b", comment)]
        if len(named) == 1:
            return named[0], set(obj.keys())
    return None


def _iter_calls_in_text(text: str) -> list[tuple[str, set[str]]]:
    """All recognised ``(tool, arg_keys)`` tool calls embedded in *text*."""
    calls: list[tuple[str, set[str]]] = []
    for block in _iter_json_blocks(text):
        for clean_obj, comment in _extract_objects(block):
            try:
                obj = json.loads(clean_obj)
            except (json.JSONDecodeError, ValueError):
                continue  # illustrative elision / response body — not validatable
            ident = _identify_call(obj, comment)
            if ident is not None:
                calls.append(ident)
    return calls


def test_ast_parses_core_tool_signatures() -> None:
    """Self-check: the AST map has the core tools and their real key params."""
    assert {"run_prompt", "run_panel", "run_quick_poll", "extend_panel"} <= REAL_TOOLS
    # pack_id landed on both panel tools (run_quick_poll via PR #561); these are
    # exactly the params the doc drift got wrong.
    assert "pack_id" in ACCEPTED_PARAMS["run_panel"]
    assert "pack_id" in ACCEPTED_PARAMS["run_quick_poll"]
    assert "questions" in ACCEPTED_PARAMS["run_panel"]
    # The invented params must NOT be accepted by any tool.
    for params in ACCEPTED_PARAMS.values():
        assert "personas_pack" not in params
        assert "stimulus" not in params


def test_guard_flags_invented_params() -> None:
    """Negative self-check: the checker actually rejects the known drift cases.

    Guards against a silently-passing guard (e.g. an extractor that matches
    nothing). Both the `stimulus` and `personas_pack` regressions are caught.
    """
    bad = """
```jsonc
// MCP tool call
{ "tool": "run_panel", "arguments": { "stimulus": "x?", "decision_being_informed": "d" } }
```

```jsonc
// MCP run_panel
{ "personas_pack": "general-consumer", "instrument_pack": "pricing-discovery" }
```
"""
    offenders = set()
    for tool, keys in _iter_calls_in_text(bad):
        offenders |= {k for k in keys if k not in ACCEPTED_PARAMS[tool]}
    assert offenders == {"stimulus", "personas_pack"}


def test_guard_examined_enough_examples() -> None:
    """Sanity floor: the extractor really found the doc tool-call examples.

    Without this a broken fence/scanner regex could make every per-file
    assertion vacuously pass.
    """
    total = sum(len(_iter_calls_in_text(f.read_text(encoding="utf-8"))) for f in _DOC_FILES if f.is_file())
    assert total >= 12, f"only {total} MCP tool-call examples parsed across the docs — the scanner is likely broken."


@pytest.mark.parametrize("doc", _DOC_FILES, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_doc_tool_call_params_are_real(doc: Path) -> None:
    """Every ``arguments`` key in a doc's MCP tool-call examples is a real param."""
    if not doc.is_file():
        pytest.skip(f"{doc} not present")
    violations: list[str] = []
    for tool, keys in _iter_calls_in_text(doc.read_text(encoding="utf-8")):
        accepted = ACCEPTED_PARAMS[tool]
        for key in sorted(keys):
            if key not in accepted:
                violations.append(f"{tool}(...): unknown parameter {key!r}")
    assert not violations, (
        f"{doc.relative_to(REPO_ROOT)} has MCP tool-call examples using parameters not in the tool's "
        f"real input schema ({SERVER_PY.relative_to(REPO_ROOT)}):\n  " + "\n  ".join(violations)
    )
