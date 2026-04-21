"""Internal utilities for Jupyter notebook display formatting."""

from __future__ import annotations

import functools
import html
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, ParamSpec, overload

P = ParamSpec("P")

__all__ = [
    "HtmlBuilder",
    "abbreviate_dict",
    "abbreviate_list",
    "render_table",
    "safe_display",
    "truncate_text",
]

_OUTER_DIV_STYLE = (
    "font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;"
    " padding: 12px; border: 1px solid #e0e0e0;"
    " border-radius: 6px; background-color: #fafafa;"
    " max-width: 500px;"
)
_TITLE_STYLE = "font-weight: 600; margin-bottom: 10px; font-size: 14px; color: #333;"
_TABLE_STYLE = "border-collapse: collapse; width: 100%;"
_LABEL_STYLE = "padding: 6px 8px; font-weight: 500; color: #666; width: 140px;"
_VALUE_STYLE = "padding: 6px 8px; text-align: left;"

_SECTION_BORDER: dict[str, str] = {
    "default": "1px solid #e0e0e0",
    "error": "1px solid #e8c4c4",
    "warning": "1px solid #f6d860",
}
_SECTION_BG: dict[str, str] = {
    "default": "#fafafa",
    "error": "#fdf2f2",
    "warning": "#fffbeb",
}
_SECTION_TITLE_COLOR: dict[str, str] = {
    "default": "#333",
    "error": "#991b1b",
    "warning": "#92400e",
}


def render_table(title: str, rows: Sequence[tuple[str, str | int | float]]) -> str:
    """Render a simple HTML table for Jupyter notebook display.

    Provides consistent styling across all model objects that implement `_repr_html_()`.

    Args:
        title: Title to display above the table
        rows: List of (label, value) tuples to render as table rows

    Returns:
        HTML string for Jupyter notebook display

    Example:
        >>> render_table("MyResponse", [("Count:", 42), ("Status:", "ok")])
        '<div style="...">...</div>'
    """
    table_rows = "\n".join(
        f"""                <tr>
                    <td style="padding: 6px 8px; font-weight: 500; color: #666;
                               width: 140px;">{html.escape(label)}</td>
                    <td style="padding: 6px 8px; text-align: left;">{html.escape(str(value))}</td>
                </tr>"""
        for label, value in rows
    )

    return f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    padding: 12px; border: 1px solid #e0e0e0;
                    border-radius: 6px; background-color: #fafafa;
                    max-width: 500px;">
            <div style="font-weight: 600; margin-bottom: 10px; font-size: 14px;
                        color: #333;">{html.escape(title)}</div>
            <table style="border-collapse: collapse; width: 100%;">
{table_rows}
            </table>
        </div>
        """


def truncate_text(value: str, max_chars: int = 80, suffix: str = "...") -> str:
    s = str(value)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + suffix


def abbreviate_list(
    items: Sequence[Any],
    head: int = 3,
    formatter: Callable[[Any], str] = repr,
) -> str:
    if len(items) == 0:
        return "[]"

    def _fmt(item: Any) -> str:
        try:
            return formatter(item)
        except Exception:
            return object.__repr__(item)

    if len(items) <= head + 2:
        return "[" + ", ".join(_fmt(x) for x in items) + "]"

    shown = ", ".join(_fmt(items[i]) for i in range(head))
    extra = len(items) - head
    return f"[{shown}, ...{extra} more]"


def abbreviate_dict(d: Mapping[str, Any], max_keys: int = 5) -> str:
    if len(d) == 0:
        return "{}"
    keys = list(d.keys())
    if len(keys) <= max_keys:
        pairs = ", ".join(f"{k}: {d[k]!r}" for k in keys)
        return "{" + pairs + "}"
    shown = ", ".join(f"{k}: {d[k]!r}" for k in keys[:max_keys])
    extra = len(keys) - max_keys
    return "{" + f"{shown}, ...{extra} more" + "}"


class HtmlBuilder:
    def __init__(self, title: str) -> None:
        self._title = html.escape(title)
        self._main_rows: list[tuple[str, str]] = []
        self._sections: list[str] = []

    def row(self, label: str, value: Any) -> HtmlBuilder:
        self._main_rows.append((html.escape(label), html.escape(str(value))))
        return self

    def rows(self, items: Iterable[tuple[str, Any]]) -> HtmlBuilder:
        for label, value in items:
            self.row(label, value)
        return self

    def section(
        self,
        title: str,
        rows: Sequence[tuple[str, Any]],
        theme: Literal["default", "error", "warning"] = "default",
    ) -> HtmlBuilder:
        border = _SECTION_BORDER[theme]
        bg = _SECTION_BG[theme]
        title_color = _SECTION_TITLE_COLOR[theme]
        escaped_title = html.escape(title)

        table_rows = "".join(
            f"""<tr>
                    <td style="{_LABEL_STYLE}">{html.escape(label)}</td>
                    <td style="{_VALUE_STYLE}">{html.escape(str(value))}</td>
                </tr>"""
            for label, value in rows
        )

        section_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    padding: 12px; border: {border};
                    border-radius: 6px; background-color: {bg};
                    max-width: 500px; margin-top: 8px;">
            <div style="font-weight: 600; margin-bottom: 10px; font-size: 14px;
                        color: {title_color};">{escaped_title}</div>
            <table style="{_TABLE_STYLE}">
                {table_rows}
            </table>
        </div>
        """
        self._sections.append(section_html)
        return self

    def build(self) -> str:
        main_rows_html = "\n".join(
            f"""                <tr>
                    <td style="{_LABEL_STYLE}">{label}</td>
                    <td style="{_VALUE_STYLE}">{value}</td>
                </tr>"""
            for label, value in self._main_rows
        )

        main_table = f"""
        <div style="{_OUTER_DIV_STYLE}">
            <div style="{_TITLE_STYLE}">{self._title}</div>
            <table style="{_TABLE_STYLE}">
{main_rows_html}
            </table>
        </div>
        """
        return main_table + "".join(self._sections)


@overload
def safe_display(method: Callable[P, str]) -> Callable[P, str]: ...


@overload
def safe_display(method: Callable[P, None]) -> Callable[P, None]: ...


def safe_display(method: Callable[..., Any]) -> Callable[..., Any]:
    is_pretty = method.__name__ == "_repr_pretty_"

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except Exception:
            self_obj = args[0] if args else None
            type_name = type(self_obj).__name__ if self_obj is not None else "Unknown"
            fallback = f"<{type_name} (display error)>"
            if is_pretty:
                p = args[1] if len(args) > 1 else None
                if p is not None:
                    p.text(fallback)
                return None
            return fallback

    return wrapper
