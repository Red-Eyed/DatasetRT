"""Reject broad type references that Ruff's banned-import rule cannot cover."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Violation:
    """Locate a forbidden reference without retaining the source tree."""

    line: int
    column: int
    name: str


def references(tree: ast.AST) -> list[Violation]:
    """Find executable references; ordinary string values and prose remain valid."""
    violations = []
    for node in ast.walk(tree):
        match node:
            case ast.Name(id=name) if name in {"object", "Any"}:
                violations.append(Violation(node.lineno, node.col_offset + 1, name))
            case ast.Attribute(attr=name) if name in {"object", "Any"}:
                violations.append(Violation(node.lineno, node.col_offset + 1, name))
    return violations


def quoted_references(annotation: ast.AST) -> list[str]:
    """Expand forward references while preserving Literal string values."""
    match annotation:
        case ast.Subscript(value=ast.Name(id="Literal")):
            return []
        case ast.Subscript(value=ast.Attribute(attr="Literal")):
            return []
        case ast.Constant(value=str() as value):
            try:
                expression = ast.parse(value, mode="eval")
            except SyntaxError:
                return []
            return [item.name for item in references(expression)] + quoted_references(expression)
        case _:
            return [
                name
                for child in ast.iter_child_nodes(annotation)
                for name in quoted_references(child)
            ]


def collect_annotations(tree: ast.AST) -> list[ast.expr]:
    """Collect annotation and cast boundaries whose strings represent types."""
    result = []
    for node in ast.walk(tree):
        match node:
            case ast.arg(annotation=annotation) | ast.AnnAssign(annotation=annotation):
                if annotation is not None:
                    result.append(annotation)
            case ast.FunctionDef(returns=annotation) | ast.AsyncFunctionDef(returns=annotation):
                if annotation is not None:
                    result.append(annotation)
            case ast.Call(func=ast.Name(id="cast"), args=[annotation, *_]):
                result.append(annotation)
            case ast.Call(func=ast.Attribute(attr="cast"), args=[annotation, *_]):
                result.append(annotation)
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            marker = node.annotation
            if (isinstance(marker, ast.Name) and marker.id == "TypeAlias") or (
                isinstance(marker, ast.Attribute) and marker.attr == "TypeAlias"
            ):
                result.append(node.value)
    return result


def check_source(source: str) -> list[Violation]:
    """Check one file without importing it or evaluating annotation expressions."""
    tree = ast.parse(source)
    # Pickle's reconstruction protocol intentionally has a heterogeneous return.
    allowed = set()
    allowed_annotations = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__reduce__" and node.returns:
            allowed.update(item for item in references(node.returns) if item.name == "object")
            allowed_annotations.add(id(node.returns))
    violations = references(tree)
    for annotation in collect_annotations(tree):
        for name in quoted_references(annotation):
            if name == "object" and id(annotation) in allowed_annotations:
                continue
            violations.append(Violation(annotation.lineno, annotation.col_offset + 1, name))
    return sorted(set(violations) - allowed, key=lambda item: (item.line, item.column, item.name))


def main() -> int:
    """Enforce the policy over the same package, test, and script roots as Ruff."""
    root = Path(__file__).resolve().parents[1]
    failed = False
    for directory in ("dataset_rt", "tests", "scripts"):
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in {".py", ".pyi"}:
                continue
            for violation in check_source(path.read_text()):
                print(
                    f"{path.relative_to(root)}:{violation.line}:{violation.column}: "
                    f"Use a precise type instead of {violation.name}"
                )
                failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
