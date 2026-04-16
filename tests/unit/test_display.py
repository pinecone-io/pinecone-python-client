"""Tests for the internal display utility module."""

from pinecone.models._display import render_table


def test_render_table_basic() -> None:
    result = render_table("Title", [("Label:", "value")])
    assert "<div" in result
    assert "Title" in result
    assert "Label:" in result
    assert "value" in result


def test_render_table_html_escape() -> None:
    result = render_table("Title", [("<script>", "<script>alert(1)</script>")])
    assert "<script>" not in result
    assert "&lt;script&gt;" in result


def test_render_table_empty_rows() -> None:
    result = render_table("Title", [])
    assert "Title" in result
    assert "<tr>" not in result


def test_render_table_numeric_values() -> None:
    result = render_table("Title", [("Int:", 42), ("Float:", 3.14)])
    assert "42" in result
    assert "3.14" in result
