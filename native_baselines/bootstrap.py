"""Fetch the exact official source revisions used by the native launchers."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from specs import SPECS

HERE = Path(__file__).resolve().parent


def command_plan(root: Path, methods):
    commands = []
    for method in methods:
        spec, dest = SPECS[method], root / method
        if not dest.exists():
            commands.append(["git", "clone", "--filter=blob:none", "--no-checkout",
                             spec["url"], str(dest)])
        commands.extend([
            ["git", "-C", str(dest), "fetch", "origin", spec["commit"], "--depth", "1"],
            ["git", "-C", str(dest), "checkout", "--detach", spec["commit"]],
        ])
    return commands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", nargs="*", metavar="METHOD")
    parser.add_argument("--root", type=Path, default=HERE / "vendor")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    methods = args.methods or list(SPECS)
    unknown = sorted(set(methods) - set(SPECS))
    if unknown:
        parser.error(f"unknown methods: {', '.join(unknown)}")
    args.root.mkdir(parents=True, exist_ok=True)
    for command in command_plan(args.root, methods):
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
