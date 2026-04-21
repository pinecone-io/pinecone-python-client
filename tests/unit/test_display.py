"""Tests for the internal display utility module."""

from __future__ import annotations

from pinecone.models._display import (
    HtmlBuilder,
    abbreviate_dict,
    abbreviate_list,
    render_table,
    safe_display,
    truncate_text,
)


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


class TestTruncateText:
    def test_returns_unchanged_below_limit(self) -> None:
        result = truncate_text("hello", max_chars=80)
        assert isinstance(result, str)
        assert result == "hello"

    def test_truncates_above_limit(self) -> None:
        value = "x" * 100
        result = truncate_text(value, max_chars=80)
        assert isinstance(result, str)
        assert result == "x" * 80 + "..."

    def test_exactly_at_limit_not_truncated(self) -> None:
        value = "y" * 80
        result = truncate_text(value, max_chars=80)
        assert isinstance(result, str)
        assert result == value

    def test_non_string_coerced(self) -> None:
        assert truncate_text(42) == "42"  # type: ignore[arg-type]
        assert truncate_text(None) == "None"  # type: ignore[arg-type]

        class Custom:
            def __str__(self) -> str:
                return "custom_str"

        assert truncate_text(Custom()) == "custom_str"  # type: ignore[arg-type]

    def test_empty_string(self) -> None:
        result = truncate_text("")
        assert isinstance(result, str)
        assert result == ""


class TestAbbreviateList:
    def test_empty_list(self) -> None:
        result = abbreviate_list([])
        assert isinstance(result, str)
        assert result == "[]"

    def test_short_list_full_render(self) -> None:
        result = abbreviate_list([1, 2, 3])
        assert isinstance(result, str)
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "more" not in result

    def test_long_list_abbreviated(self) -> None:
        items = list(range(1536))
        result = abbreviate_list(items)
        assert isinstance(result, str)
        assert "1533 more" in result
        assert result.startswith("[")

    def test_custom_formatter(self) -> None:
        result = abbreviate_list([1, 2, 3, 4, 5, 6], formatter=str)
        assert isinstance(result, str)
        assert "1" in result

    def test_formatter_exception_falls_back(self) -> None:
        def bad_formatter(x: object) -> str:
            raise ValueError("boom")

        items = list(range(10))
        result = abbreviate_list(items, formatter=bad_formatter)
        assert isinstance(result, str)
        assert "more" in result

    def test_head_boundary(self) -> None:
        # exactly head+2=5 items should be rendered full (no "more")
        items = [1, 2, 3, 4, 5]
        result = abbreviate_list(items, head=3)
        assert isinstance(result, str)
        assert "more" not in result
        # 6 items should be abbreviated
        items6 = [1, 2, 3, 4, 5, 6]
        result6 = abbreviate_list(items6, head=3)
        assert "more" in result6


class TestAbbreviateDict:
    def test_empty_dict(self) -> None:
        result = abbreviate_dict({})
        assert isinstance(result, str)
        assert result == "{}"

    def test_small_dict_full(self) -> None:
        d = {"a": 1, "b": 2}
        result = abbreviate_dict(d)
        assert isinstance(result, str)
        assert "a" in result
        assert "b" in result
        assert "more" not in result

    def test_large_dict_abbreviated(self) -> None:
        d = {f"key{i}": i for i in range(10)}
        result = abbreviate_dict(d, max_keys=5)
        assert isinstance(result, str)
        assert "more" in result
        assert "5 more" in result


class TestHtmlBuilder:
    def test_title_only(self) -> None:
        result = HtmlBuilder("MyModel").build()
        assert isinstance(result, str)
        assert "MyModel" in result
        assert "<div" in result

    def test_rows_and_sections(self) -> None:
        result = (
            HtmlBuilder("Test").row("Field:", "value").section("Extra", [("Key:", "val")]).build()
        )
        assert isinstance(result, str)
        assert "Field:" in result
        assert "value" in result
        assert "Extra" in result
        assert "Key:" in result

    def test_html_escape_in_label_value_section_title(self) -> None:
        result = (
            HtmlBuilder("<Title>")
            .row("<label>", "<value>")
            .section("<section>", [("<k>", "<v>")])
            .build()
        )
        assert isinstance(result, str)
        assert "&lt;Title&gt;" in result
        assert "&lt;label&gt;" in result
        assert "&lt;value&gt;" in result
        assert "&lt;section&gt;" in result

    def test_error_theme_uses_red_palette(self) -> None:
        result = HtmlBuilder("Test").section("Errors", [("msg:", "bad")], theme="error").build()
        assert isinstance(result, str)
        assert "#991b1b" in result

    def test_warning_theme_uses_amber_palette(self) -> None:
        result = (
            HtmlBuilder("Test").section("Warnings", [("msg:", "caution")], theme="warning").build()
        )
        assert isinstance(result, str)
        assert "#92400e" in result or "#fffbeb" in result

    def test_empty_section_still_renders(self) -> None:
        result = HtmlBuilder("Test").section("Empty", []).build()
        assert isinstance(result, str)
        assert "Empty" in result

    def test_value_with_none(self) -> None:
        result = HtmlBuilder("Test").row("Field:", None).build()
        assert isinstance(result, str)
        assert "None" in result


class TestSafeDisplay:
    def test_passes_through_normal_return(self) -> None:
        class Obj:
            @safe_display
            def __repr__(self) -> str:
                return "normal"

        obj = Obj()
        result = repr(obj)
        assert result == "normal"

    def test_catches_exception_from_repr(self) -> None:
        class Obj:
            @safe_display
            def __repr__(self) -> str:
                raise RuntimeError("boom")

        obj = Obj()
        result = repr(obj)
        assert isinstance(result, str)
        assert "Obj" in result
        assert "display error" in result

    def test_catches_exception_from_repr_html(self) -> None:
        class Obj:
            @safe_display
            def _repr_html_(self) -> str:
                raise ValueError("fail")

        obj = Obj()
        result = obj._repr_html_()
        assert isinstance(result, str)
        assert "Obj" in result
        assert "display error" in result

    def test_catches_exception_from_repr_pretty(self) -> None:
        class FakePretty:
            def __init__(self) -> None:
                self.written: str = ""

            def text(self, s: str) -> None:
                self.written = s

        class Obj:
            @safe_display
            def _repr_pretty_(self, p: FakePretty, cycle: bool) -> None:
                raise AttributeError("no attribute")

        obj = Obj()
        p = FakePretty()
        result = obj._repr_pretty_(p, False)
        assert result is None
        assert "display error" in p.written

    def test_preserves_method_name_via_functools_wraps(self) -> None:
        class Obj:
            @safe_display
            def __repr__(self) -> str:
                return "ok"

        obj = Obj()
        assert obj.__repr__.__name__ == "__repr__"

    def test_nested_attribute_error_does_not_propagate(self) -> None:
        class Obj:
            @safe_display
            def __repr__(self) -> str:
                _ = self.nonexistent_attr  # type: ignore[attr-defined]
                return "unreachable"

        obj = Obj()
        result = repr(obj)
        assert isinstance(result, str)
        assert "display error" in result
