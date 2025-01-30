import pytest
import re


def use_in_match(headers):
    ct = headers.get("Content-Type", "")
    return "json" in ct


def use_re_match(headers):
    ct = headers.get("Content-Type", "")
    return re.search("json", ct, re.IGNORECASE)


class TestStringMatchPerformance:
    @pytest.mark.parametrize(
        "headers",
        [
            {"Content-Type": "application/json"}
            # {"Content-Type": "application/JSON"},
            # {"Content-Type": "application/xml"},
            # {"Content-Type": "text/html"},
            # {"Content-Type": "text/plain"},
            # {"Content-Type": "application/xml"},
            # {"Content-Type": "text/html"},
            # {"Content-Type": "text/plain"},
        ],
    )
    def test_using_in(self, benchmark, headers):
        benchmark.pedantic(use_in_match, (headers,), rounds=100, warmup_rounds=1, iterations=50)

    @pytest.mark.parametrize(
        "headers",
        [
            {"Content-Type": "application/json"}
            # {"Content-Type": "application/JSON"},
            # {"Content-Type": "application/xml"},
            # {"Content-Type": "text/html"},
            # {"Content-Type": "text/plain"},
            # {"Content-Type": "application/xml"},
            # {"Content-Type": "text/html"},
            # {"Content-Type": "text/plain"},
        ],
    )
    def test_using_re(self, benchmark, headers):
        benchmark.pedantic(use_re_match, (headers,), rounds=100, warmup_rounds=1, iterations=50)
