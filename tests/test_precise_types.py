"""Verify type-policy enforcement without rejecting ordinary prose or literals."""

import pytest

from scripts.check_precise_types import check_source


@pytest.mark.parametrize(
    "source",
    [
        "x: object",
        "x: typing.Any",
        "x = object()",
        'x: "list[object]"',
        'def f(x: "Any") -> None: pass',
        'x = cast("tuple[object, ...]", value)',
        'Alias: TypeAlias = "object"',
        "x = builtins.object()",
        "def __reduce__(self) -> Any: pass",
        "def __reduce__(self): return object()",
    ],
)
def test_rejects_broad_types(source: str) -> None:
    """Cover executable names and quoted types that banned imports can miss."""
    assert check_source(source)


@pytest.mark.parametrize(
    "source",
    [
        '"""An object returned by the API."""',
        'message = "object"',
        'x: Literal["object", "Any"]',
        "x: \"typing.Literal['object']\"",
        'x: "list[int]"',
        'x: "tuple[ReaderRecipe, int]"',
        "def __reduce__(self) -> tuple[object, tuple[object, ...]]: pass",
        'def __reduce__(self) -> "tuple[object, ...]": pass',
    ],
)
def test_allows_precise_types_and_prose(source: str) -> None:
    """Keep documentation and literal domain values outside the type ban."""
    assert not check_source(source)
