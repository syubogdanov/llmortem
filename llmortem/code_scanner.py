from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List

from llmortem.document_loader import safe_read_text


PY_COMMENT_RE = re.compile(r"^\s*#\s?(.*)$")
GENERIC_COMMENT_RE = re.compile(r"^\s*(#|//|--)\s?(.*)$")


def extract_python_knowledge(path: Path) -> str:
    raw = safe_read_text(path)
    blocks: List[str] = []

    try:
        tree = ast.parse(raw)
    except SyntaxError:
        return extract_generic_code_knowledge(path)

    module_doc = ast.get_docstring(tree)
    if module_doc:
        blocks.append(f"Module docstring:\n{module_doc}")

    lines = raw.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = [arg.arg for arg in node.args.args]
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                signature = f"{prefix}def {node.name}({', '.join(args)})"
            else:
                signature = f"class {node.name}"

            if doc:
                blocks.append(f"{signature} at line {getattr(node, 'lineno', '?')}:\n{doc}")
            else:
                blocks.append(f"{signature} at line {getattr(node, 'lineno', '?')}")

    comment_runs: List[str] = []
    current: List[str] = []

    for line in lines:
        match = PY_COMMENT_RE.match(line)
        if match:
            current.append(match.group(1))
        else:
            if current and len(" ".join(current).strip()) > 20:
                comment_runs.append("\n".join(current))
            current = []

    if current and len(" ".join(current).strip()) > 20:
        comment_runs.append("\n".join(current))

    for index, comment in enumerate(comment_runs[:80], start=1):
        blocks.append(f"Comment block {index}:\n{comment}")

    return "\n\n".join(blocks)


def extract_generic_code_knowledge(path: Path) -> str:
    raw = safe_read_text(path)
    lines = raw.splitlines()
    blocks: List[str] = []
    current: List[str] = []

    declaration_patterns = [
        re.compile(r"\bfunction\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bclass\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bfunc\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bdef\s+([A-Za-z0-9_]+)"),
        re.compile(r"\bpublic\s+.*\s+([A-Za-z0-9_]+)\("),
    ]

    for line_number, line in enumerate(lines, start=1):
        comment = GENERIC_COMMENT_RE.match(line)

        if comment:
            current.append(comment.group(2))
            continue

        if current:
            text = "\n".join(current).strip()
            if len(text) > 20:
                blocks.append(f"Comment block near line {max(1, line_number - len(current))}:\n{text}")
            current = []

        for pattern in declaration_patterns:
            if pattern.search(line):
                blocks.append(f"Declaration near line {line_number}: {line.strip()}")
                break

    if current:
        text = "\n".join(current).strip()
        if len(text) > 20:
            blocks.append(f"Comment block near EOF:\n{text}")

    return "\n\n".join(blocks)