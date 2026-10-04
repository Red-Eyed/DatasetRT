"""Generate API documentation statically from definitions behind public exports."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = PROJECT_ROOT / "dataset_rt" / "api.py"
PACKAGE_SOURCE = PROJECT_ROOT / "dataset_rt" / "__init__.py"
PYTHON_API_DOC = PROJECT_ROOT / "docs" / "python-api.md"

BEGIN_MARKER = "<!-- BEGIN GENERATED: Public Python API -->"
END_MARKER = "<!-- END GENERATED: Public Python API -->"
SPECIAL_METHODS = {"__init__", "__iter__", "__len__"}


@dataclass(frozen=True)
class ApiDefinition:
    """Keep an AST definition paired with its source for alias rendering."""

    source: str
    node: ast.AST


def main() -> None:
    """Resolve facade exports statically so optional dependencies stay unloaded."""
    check = parse_check_flag()
    package_module = ast.parse(PACKAGE_SOURCE.read_text())
    generated = render_generated_section(public_names(package_module))
    update_marked_section(PYTHON_API_DOC, generated, check=check)


def parse_check_flag() -> bool:
    args = sys.argv[1:]
    if args == ["--check"]:
        return True
    if args:
        raise SystemExit("usage: sync_api_docs.py [--check]")
    return False


def public_names(module: ast.Module) -> list[str]:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return literal_string_list(node.value)
    raise SystemExit("__all__ not found")


def literal_string_list(node: ast.AST) -> list[str]:
    if not isinstance(node, ast.List):
        raise SystemExit("__all__ must be a list literal")
    names = []
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise SystemExit("__all__ entries must be string literals")
        names.append(element.value)
    return names


def render_generated_section(exported_names: list[str]) -> str:
    """Render definitions behind the facade in the package's declared export order."""
    sections = [
        BEGIN_MARKER,
        "",
        "_Generated from public docstrings behind the `dataset_rt.api` facade._",
        "",
    ]
    objects = facade_definitions()
    for name in exported_names:
        definition = objects.get(name)
        if definition is None:
            raise SystemExit(f"exported object has no API definition: {name}")
        sections.extend(render_public_object(definition.source, name, definition.node))
        sections.append("")
    sections.append(END_MARKER)
    return "\n".join(sections)


def facade_definitions() -> dict[str, ApiDefinition]:
    """Follow explicit implementation imports without executing package code."""
    facade = ast.parse(API_SOURCE.read_text())
    definitions: dict[str, ApiDefinition] = {}
    modules: dict[str, dict[str, ApiDefinition]] = {}
    for node in facade.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not node.module.startswith("dataset_rt."):
            continue
        if node.module not in modules:
            path = PROJECT_ROOT.joinpath(*node.module.split(".")).with_suffix(".py")
            source = path.read_text()
            module = ast.parse(source)
            modules[node.module] = {
                name: ApiDefinition(source, definition)
                for name, definition in public_api_objects(module).items()
            }
        for alias in node.names:
            definition = modules[node.module].get(alias.name)
            if definition is not None:
                definitions[alias.asname or alias.name] = definition
    return definitions


def public_api_objects(module: ast.Module) -> dict[str, ast.AST]:
    objects: dict[str, ast.AST] = {}
    for node in module.body:
        name = public_node_name(node)
        if name is not None:
            objects[name] = node
    return objects


def public_node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.ClassDef):
        return node.name
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None


def render_public_object(source: str, name: str, node: ast.AST) -> list[str]:
    if isinstance(node, ast.ClassDef):
        return render_class(name, node)
    if isinstance(node, ast.AnnAssign):
        return render_type_alias(source, name, node)
    raise SystemExit(f"unsupported public API node: {name}")


def render_type_alias(source: str, name: str, node: ast.AnnAssign) -> list[str]:
    """Require an initialized alias and its adjacent source documentation."""
    if node.value is None:
        raise SystemExit(f"type alias value not found: {name}")
    annotation = ast.get_source_segment(source, node.value)
    if annotation is None:
        raise SystemExit(f"type alias source not found: {name}")
    return [
        f"### `{name}`",
        "",
        f"```python\n{name} = {annotation}\n```",
        "",
        node_docstring(source, name, node),
    ]


def render_class(name: str, class_def: ast.ClassDef) -> list[str]:
    sections = [f"### `{name}`", "", format_docstring(class_def)]
    fields = class_fields(class_def)
    if fields:
        sections.append("")
        sections.append("Fields:")
        sections.extend(f"- `{field}`: {doc}" for field, doc in fields)
    methods = class_methods(class_def)
    for method in methods:
        sections.append("")
        if is_property(method):
            sections.append(f"#### `{name}.{method.name}`")
        else:
            sections.append(f"#### `{name}.{method_signature(method)}`")
        sections.append("")
        sections.append(format_docstring(method))
    return sections


def class_fields(class_def: ast.ClassDef) -> list[tuple[str, str]]:
    fields: list[tuple[str, str]] = []
    body = class_def.body
    for index, node in enumerate(body):
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id == "model_config" or node.target.id.startswith("_"):
            continue
        doc = field_description(node) or following_docstring(body, index)
        if doc is None:
            raise SystemExit(f"field is missing docstring: {class_def.name}.{node.target.id}")
        annotation = ast.unparse(node.annotation)
        fields.append((f"{node.target.id}: {annotation}", doc))
    return fields


def field_description(node: ast.AnnAssign) -> str | None:
    if not isinstance(node.value, ast.Call):
        return None
    if not isinstance(node.value.func, ast.Name) or node.value.func.id != "Field":
        return None
    for keyword in node.value.keywords:
        if keyword.arg != "description":
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return inspect.cleandoc(keyword.value.value)
    return None


def class_methods(class_def: ast.ClassDef) -> list[ast.FunctionDef]:
    return [
        node
        for node in class_def.body
        if isinstance(node, ast.FunctionDef) and is_documented_method(node)
    ]


def is_documented_method(method: ast.FunctionDef) -> bool:
    return not method.name.startswith("_") or method.name in SPECIAL_METHODS


def is_property(method: ast.FunctionDef) -> bool:
    return any(
        isinstance(decorator, ast.Name) and decorator.id == "property"
        for decorator in method.decorator_list
    )


def method_signature(method: ast.FunctionDef) -> str:
    rendered_args = render_args(method.args)
    returns = ""
    if method.name != "__init__" and method.returns is not None:
        returns = f" -> {ast.unparse(method.returns)}"
    return f"{method.name}({rendered_args}){returns}"


def render_args(args_node: ast.arguments) -> str:
    args = list(args_node.posonlyargs) + list(args_node.args)
    if args and args[0].arg == "self":
        args = args[1:]
    defaults: list[ast.expr | None] = [None] * (len(args) - len(args_node.defaults))
    defaults.extend(args_node.defaults)
    rendered = [render_arg(arg, default) for arg, default in zip(args, defaults, strict=True)]
    if args_node.kwonlyargs:
        rendered.append("*")
        rendered.extend(
            render_arg(arg, default)
            for arg, default in zip(args_node.kwonlyargs, args_node.kw_defaults, strict=True)
        )
    return ", ".join(rendered)


def render_arg(arg: ast.arg, default: ast.expr | None) -> str:
    if arg.annotation is None:
        rendered = arg.arg
    else:
        rendered = f"{arg.arg}: {ast.unparse(arg.annotation)}"
    if default is not None:
        rendered = f"{rendered} = {ast.unparse(default)}"
    return rendered


def format_docstring(node: ast.ClassDef | ast.FunctionDef) -> str:
    docstring = ast.get_docstring(node)
    if docstring is None:
        raise SystemExit(f"object is missing docstring: {node.name}")
    return inspect.cleandoc(docstring)


def node_docstring(source: str, name: str, node: ast.AnnAssign) -> str:
    """Resolve alias documentation without attaching untyped fields to AST nodes."""
    body = ast.parse(source).body
    for index, statement in enumerate(body):
        if statement.lineno != node.lineno:
            continue
        doc = following_docstring(body, index)
        if doc is not None:
            return doc
        break
    raise SystemExit(f"type alias is missing docstring: {name}")


def following_docstring(body: list[ast.stmt], index: int) -> str | None:
    next_index = index + 1
    if next_index >= len(body):
        return None
    node = body[next_index]
    if not isinstance(node, ast.Expr):
        return None
    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
        return None
    return inspect.cleandoc(node.value.value)


def update_marked_section(path: Path, generated: str, *, check: bool) -> None:
    original = path.read_text()
    start = original.find(BEGIN_MARKER)
    end = original.find(END_MARKER)
    if start == -1 or end == -1:
        raise SystemExit(f"generated API docs markers not found in {path}")
    end += len(END_MARKER)
    updated = f"{original[:start]}{generated}{original[end:]}"
    if check and updated != original:
        raise SystemExit(f"{path} is not synchronized with API docstrings")
    if check:
        return
    path.write_text(updated)


if __name__ == "__main__":
    main()
