"""Internal utilities for Jupyter notebook display formatting."""

import html

from collections.abc import Sequence


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
