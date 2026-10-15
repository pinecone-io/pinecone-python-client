"""The package metadata PyPI renders must send readers to the SDK reference.

``Documentation`` in ``[project.urls]`` is what the PyPI sidebar and
``importlib.metadata`` hand to anyone looking for SDK docs. The guides that
docstrings link to live under ``https://sdk.pinecone.io/python/``, so that is
where it has to point. The check reads the ``METADATA`` file the build backend
generates, not ``pyproject.toml``, because the metadata is what ships.

Generating it runs ``maturin pep517 write-dist-info`` through ``uv tool run``,
using the same version range as ``[build-system].requires``. That needs ``uv``,
``cargo`` and, on a cold cache, network access; it compiles nothing.
"""

from __future__ import annotations

import shutil
import subprocess
from email.parser import Parser
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

DOCUMENTATION_URL = "https://sdk.pinecone.io/python/"


@pytest.fixture(scope="module")
def project_urls(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not on PATH; it runs the build backend that generates the metadata")
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not on PATH; maturin needs it to read the crate metadata")
    out = tmp_path_factory.mktemp("dist-info")
    proc = subprocess.run(  # noqa: S603 — fixed argv, uv resolved via shutil.which
        [
            uv,
            "tool",
            "run",
            "--from",
            "maturin>=1.0,<2.0",
            "maturin",
            "pep517",
            "write-dist-info",
            "--metadata-directory",
            str(out),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"maturin write-dist-info failed:\n{proc.stderr[-2000:]}"
    found = sorted(out.glob("*.dist-info/METADATA"))
    assert len(found) == 1, f"expected exactly one METADATA file, found {found}"
    message = Parser().parsestr(found[0].read_text(encoding="utf-8"))
    urls: dict[str, str] = {}
    for entry in message.get_all("Project-URL") or []:
        label, _, url = entry.partition(", ")
        urls[label] = url
    return urls


def test_documentation_url_points_at_sdk_reference(project_urls: dict[str, str]) -> None:
    assert project_urls.get("Documentation") == DOCUMENTATION_URL, (
        f"[project.urls] Documentation in pyproject.toml must be {DOCUMENTATION_URL}; "
        f"labels present: {sorted(project_urls)}"
    )


def test_changelog_url_shares_the_sdk_reference_site(project_urls: dict[str, str]) -> None:
    assert project_urls.get("Changelog", "").startswith(DOCUMENTATION_URL)
