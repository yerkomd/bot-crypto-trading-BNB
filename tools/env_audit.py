#!/usr/bin/env python3
"""Env audit tool.

Scans the codebase for environment variables (os.getenv, _env_* helpers)
and compares them to the variables defined in `.env.example`.

Usage:
  python tools/env_audit.py
  python tools/env_audit.py --root . --env-file .env.example

Exit code:
  0 always (informational). Designed to be safe to run locally/CI.
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable


_GETENV_RE = re.compile(r"os\.getenv\(\s*([\"'])(?P<name>[A-Z0-9_]+)\1")
_ENVIRON_GET_RE = re.compile(r"os\.environ\.get\(\s*([\"'])(?P<name>[A-Z0-9_]+)\1")
# Match any project helper like _env_int/_env_float/_env_list/_env_datetime/etc.
_ENV_HELPER_RE = re.compile(r"\b_env_[A-Za-z0-9_]*\(\s*([\"'])(?P<name>[A-Z0-9_]+)\1")


def _iter_py_files(root: Path, *, include_legacy: bool) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        # Avoid scanning virtualenvs / caches / vendored libs
        parts = {x.lower() for x in p.parts}
        if (
            ".venv" in parts
            or "venv" in parts
            or "__pycache__" in parts
            or ".pytest_cache" in parts
            or ".tox" in parts
            or ".mypy_cache" in parts
            or "site-packages" in parts
            or "dist-packages" in parts
            or ".env-bot-trading" in parts
        ):
            continue

        if not include_legacy:
            # Keep the report focused on the current institutional bot (v3.1) + services.
            legacy_names = {
                "bot-trading.py",
                "bot_trading_v2.py",
                "bot_trading_v2_2.py",
                "bot_trading_v3.py",
            }
            if p.name in legacy_names:
                continue
        yield p


def _parse_env_example(env_file: Path) -> set[str]:
    vars_found: set[str] = set()
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Handle: export FOO=bar
        if s.startswith("export "):
            s = s[len("export ") :].lstrip()
        if "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        if re.fullmatch(r"[A-Z0-9_]+", key):
            vars_found.add(key)
    return vars_found


def _scan_code_for_env_vars(root: Path, *, include_legacy: bool) -> DefaultDict[str, set[str]]:
    used: DefaultDict[str, set[str]] = defaultdict(set)
    for f in _iter_py_files(root, include_legacy=include_legacy):
        try:
            txt = f.read_text(encoding="utf-8")
        except Exception:
            continue

        for m in _GETENV_RE.finditer(txt):
            used[m.group("name")].add(str(f.relative_to(root)))
        for m in _ENVIRON_GET_RE.finditer(txt):
            used[m.group("name")].add(str(f.relative_to(root)))
        for m in _ENV_HELPER_RE.finditer(txt):
            used[m.group("name")].add(str(f.relative_to(root)))

    return used


def _print_section(title: str, items: list[str]) -> None:
    print(f"\n== {title} ({len(items)}) ==")
    for s in items:
        print(s)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root to scan")
    ap.add_argument("--env-file", default=".env.example", help="Env example file")
    ap.add_argument(
        "--include-legacy",
        action="store_true",
        help="Include legacy scripts (bot_trading_v2/v3 and bot-trading.py) in the scan",
    )
    ap.add_argument(
        "--show-files",
        action="store_true",
        help="Show file locations for each used env var",
    )

    args = ap.parse_args()

    root = Path(args.root).resolve()
    env_file = (root / args.env_file).resolve()

    if not env_file.exists():
        raise SystemExit(f"Env file not found: {env_file}")

    defined = _parse_env_example(env_file)
    used_map = _scan_code_for_env_vars(root, include_legacy=bool(args.include_legacy))
    used = set(used_map.keys())

    missing = sorted(used - defined)
    extra = sorted(defined - used)

    # Some env vars are intentionally present for external tooling / manual overrides.
    # Keep this allowlist small and explicit.
    allow_extra = {
        # none for now
    }
    extra_effective = [x for x in extra if x not in allow_extra]

    print(f"Root: {root}")
    print(f"Env file: {env_file}")
    print(f"Defined in env example: {len(defined)}")
    print(f"Used in code: {len(used)}")

    if missing:
        _print_section("USED BUT MISSING IN .env.example", [f"- {k}" for k in missing])
    else:
        _print_section("USED BUT MISSING IN .env.example", ["(none)"])

    if extra_effective:
        _print_section("DEFINED IN .env.example BUT NOT USED IN CODE", [f"- {k}" for k in extra_effective])
    else:
        _print_section("DEFINED IN .env.example BUT NOT USED IN CODE", ["(none)"])

    if args.show_files and used_map:
        print("\n== USAGES ==")
        for k in sorted(used_map.keys()):
            files = sorted(used_map[k])
            files_s = ", ".join(files)
            print(f"{k}: {files_s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
