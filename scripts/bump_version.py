#!/usr/bin/env python3
"""Compute (and optionally apply) a release version bump.

The script is the single source of truth for "given the current
pyproject.toml version, plus a bump level and a PEP 440 prerelease suffix,
what version are we publishing?"

It emits three values for callers to use:

    pep440_version  e.g. 9.0.0rc1   -> goes into pyproject.toml + wheel name
    cargo_version   e.g. 9.0.0-rc1  -> goes into rust/Cargo.toml
    tag_name        e.g. v9.0.0rc1  -> the git tag we push

Without ``--write`` the script just prints these (and writes them to
``$GITHUB_OUTPUT`` if set). With ``--write`` it patches pyproject.toml +
rust/Cargo.toml in-place, and (unless ``--no-cargo-lock``) regenerates
rust/Cargo.lock so it matches the new version.

Usage examples:

    # Compute only (current pyproject.toml version is 9.0.0)
    python scripts/bump_version.py --suffix rc1
        -> 9.0.0rc1

    python scripts/bump_version.py --level minor --suffix rc1
        -> 9.1.0rc1

    python scripts/bump_version.py --level patch
        -> 9.0.1   (no suffix => final-release form)

    # Compute + write
    python scripts/bump_version.py --level none --suffix rc2 --write
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import subprocess
import sys

try:
    import tomllib  # py3.11+
except ImportError:  # pragma: no cover  -- only hit on 3.10
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CARGO_TOML = REPO_ROOT / "rust" / "Cargo.toml"
PINECONE_INIT = REPO_ROOT / "pinecone" / "__init__.py"
DOCS_CONF = REPO_ROOT / "docs" / "conf.py"

_BASE_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)")
_SUFFIX_RE = re.compile(r"^(a|b|rc)(\d+)$")
_PEP440_FULL_RE = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")

# `version = "..."` line in TOML files (pyproject.toml / Cargo.toml).
_TOML_VERSION_RE = re.compile(
    r'^(?P<prefix>\s*version\s*=\s*")(?P<value>[^"]+)(?P<suffix>")',
    flags=re.MULTILINE,
)
# `__version__ = "..."` line in pinecone/__init__.py.
_PY_VERSION_RE = re.compile(
    r'^(?P<prefix>__version__\s*=\s*")(?P<value>[^"]+)(?P<suffix>")',
    flags=re.MULTILINE,
)
# `release = "..."` line in Sphinx docs/conf.py.
_DOCS_RELEASE_RE = re.compile(
    r'^(?P<prefix>release\s*=\s*")(?P<value>[^"]+)(?P<suffix>")',
    flags=re.MULTILINE,
)


def parse_base(version: str) -> tuple[int, int, int]:
    m = _BASE_VERSION_RE.match(version)
    if not m:
        raise ValueError(f"unrecognised base version: {version!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def bump_base(base: tuple[int, int, int], level: str) -> tuple[int, int, int]:
    major, minor, patch = base
    if level == "none":
        return base
    if level == "patch":
        return major, minor, patch + 1
    if level == "minor":
        return major, minor + 1, 0
    if level == "major":
        return major + 1, 0, 0
    raise ValueError(f"unknown level: {level!r} (expected none|patch|minor|major)")


def normalise_suffix(suffix: str) -> str:
    """Accept rc1, rc.1, rc-1, RC1 -> normalise to PEP 440 (e.g. 'rc1'); '' -> ''."""
    if not suffix:
        return ""
    s = suffix.strip().lower().replace(".", "").replace("-", "")
    if not _SUFFIX_RE.match(s):
        raise ValueError(f"prerelease suffix must match aN|bN|rcN (got {suffix!r})")
    return s


def to_cargo(pep440: str) -> str:
    """Convert a PEP 440 version to a SemVer string Cargo accepts.

    9.0.0    -> 9.0.0
    9.0.0rc1 -> 9.0.0-rc1
    """
    m = re.match(r"^(\d+\.\d+\.\d+)(.*)$", pep440)
    if not m:
        raise ValueError(f"can't convert to cargo: {pep440!r}")
    base, rest = m.group(1), m.group(2)
    return base if not rest else f"{base}-{rest}"


def read_current() -> str:
    with PYPROJECT.open("rb") as f:
        return tomllib.load(f)["project"]["version"]


def patch_line(path: pathlib.Path, pattern: re.Pattern[str], new_value: str) -> None:
    """Replace the *first* match of ``pattern`` in ``path`` with ``new_value``."""
    text = path.read_text()
    new_text, n = pattern.subn(
        lambda m: f"{m.group('prefix')}{new_value}{m.group('suffix')}",
        text,
        count=1,
    )
    if n != 1:
        sys.exit(f"error: failed to patch version line in {path}")
    path.write_text(new_text)


def write_files(pep440: str, cargo: str, *, run_cargo_lock: bool) -> None:
    """Patch every file that carries a hardcoded version of the SDK.

    Add new files here whenever a new copy of the version string sneaks in.
    """
    patch_line(PYPROJECT, _TOML_VERSION_RE, pep440)
    patch_line(CARGO_TOML, _TOML_VERSION_RE, cargo)
    patch_line(PINECONE_INIT, _PY_VERSION_RE, pep440)
    patch_line(DOCS_CONF, _DOCS_RELEASE_RE, pep440)
    if run_cargo_lock:
        # `cargo update --workspace` only refreshes workspace-member entries in
        # Cargo.lock; transitive deps stay pinned. This avoids the multi-hundred-
        # line lockfile churn that `cargo generate-lockfile` produces.
        subprocess.run(
            ["cargo", "update", "--workspace", "--manifest-path", str(CARGO_TOML)],
            check=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--level",
        choices=["none", "patch", "minor", "major"],
        default="none",
        help="bump the X.Y.Z component before appending the suffix (default: none)",
    )
    ap.add_argument(
        "--suffix",
        default="",
        help="PEP 440 prerelease suffix (e.g. 'rc1'); empty => final release form",
    )
    ap.add_argument(
        "--current",
        default=None,
        help="override the 'current version' input (default: read pyproject.toml)",
    )
    ap.add_argument(
        "--set",
        dest="explicit",
        default=None,
        help="set the version explicitly (PEP 440); ignores --level/--suffix/--current",
    )
    ap.add_argument(
        "--write",
        action="store_true",
        help="patch every file that carries a hardcoded version (see write_files)",
    )
    ap.add_argument(
        "--no-cargo-lock",
        action="store_true",
        help="skip cargo generate-lockfile when --write is set",
    )
    args = ap.parse_args()

    if args.explicit is not None:
        if not _PEP440_FULL_RE.match(args.explicit):
            sys.exit(f"error: --set value must match X.Y.Z[aN|bN|rcN] (got {args.explicit!r})")
        pep440 = args.explicit
        current = args.current or read_current()
    else:
        current = args.current or read_current()
        new_base = bump_base(parse_base(current), args.level)
        suffix = normalise_suffix(args.suffix)
        pep440 = f"{new_base[0]}.{new_base[1]}.{new_base[2]}{suffix}"

    cargo = to_cargo(pep440)
    tag = f"v{pep440}"

    if args.write:
        write_files(pep440, cargo, run_cargo_lock=not args.no_cargo_lock)

    lines = [
        f"current_version={current}",
        f"pep440_version={pep440}",
        f"cargo_version={cargo}",
        f"tag_name={tag}",
    ]
    out_path = os.environ.get("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a") as f:
            f.write("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
