#!/usr/bin/env python3

"""
Cross-platform docs/snippets check runner.

Replicates docs_check/check (bash) behavior for Windows + MSVC and other platforms.
Requires Python 3.7+ and an available compiler (cl, g++, or clang++).

Usage: python docs_check/check.py
"""

from __future__ import annotations

import atexit
import difflib
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def print_flush(*args: object, **kwargs: Any) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def run_inner_check(snipdir: Path, include_dir: Path) -> tuple[int, str]:
    md_files = sorted(
        glob.glob(str(repo_root() / "docs" / "**" / "*.md"), recursive=True)
    )
    md_files = [str(Path(f).relative_to(repo_root())) for f in md_files]
    if not md_files:
        return 1, "No markdown files found under docs/"

    cmd = [
        sys.executable,
        str(repo_root() / "docs_check" / "inner-check.py"),
        str(snipdir),
        str(include_dir),
        *md_files,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def compare_expected(output: str) -> bool:
    expected_path = repo_root() / "docs_check" / "expected-warnings"
    try:
        expected = expected_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print_flush("Expected warnings file not found:", expected_path)
        return False

    # Normalize newlines
    def norm(s: str) -> str:
        return s.replace("\r\n", "\n").replace("\r", "\n")

    exp_n = norm(expected).rstrip("\n") + "\n"
    out_n = norm(output).rstrip("\n") + "\n"

    if exp_n == out_n:
        print_flush("Markdown links and formatting OK")
        return True

    print_flush("Unexpected Markdown warnings (diff vs expected):")
    diff = difflib.unified_diff(
        exp_n.splitlines(keepends=True),
        out_n.splitlines(keepends=True),
        fromfile=str(expected_path),
        tofile="<actual>",
    )
    sys.stdout.writelines(diff)
    sys.stdout.flush()

    if os.environ.get("CI"):
        # In CI, mismatch is fatal
        sys.exit(1)

    return False


def which(name: str) -> str | None:
    return shutil.which(name)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print_flush("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def compile_with_msvc(snipdir: Path, include: Path, include_dummy: Path) -> None:
    cl = which("cl")
    if not cl:
        return

    hpps = sorted(snipdir.glob("*.hpp"))
    cpps = sorted(snipdir.glob("*.cpp"))

    print_flush("MSVC, C++20...")
    if hpps:
        run(
            [
                cl,
                "/nologo",
                "/Zs",
                "/std:c++20",
                "/O2",
                "/W4",
                "/EHsc",
                "/permissive-",
                "/wd4100",  # unreferenced parameter
                "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
                "/wd4189",  # local variable initialized but not referenced"
                "/wd4244",  # conversion from 'type1' to 'type2', possible loss of data
                "/wd4456",  # declaration of 'x' hides previous local declaration"
                "/wd4505",  # unreferenced local function has been removed"
                f"/I{include}",
                f"/I{include_dummy}",
                "/TP",
                *[str(h) for h in hpps],
            ]
        )

    if cpps:
        exe = snipdir / "a.exe"
        # Clean previous outputs to avoid stale link artifacts
        for p in [exe, *snipdir.glob("*.obj")]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        run(
            [
                cl,
                "/nologo",
                "/std:c++20",
                "/O2",
                "/W4",
                "/EHsc",
                "/permissive-",
                "/wd4100",  # unreferenced parameter
                "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
                "/wd4189",  # local variable initialized but not referenced"
                "/wd4244",  # conversion from 'type1' to 'type2', possible loss of data
                "/wd4456",  # declaration of 'x' hides previous local declaration"
                "/wd4505",  # unreferenced local function has been removed"
                f"/I{include}",
                f"/I{include_dummy}",
                "/TP",
                *[str(c) for c in cpps],
                f"/Fe:{exe}",
            ]
        )
        run([str(exe)])

    print_flush("MSVC, C++latest...")
    if hpps:
        run(
            [
                cl,
                "/nologo",
                "/Zs",
                "/std:c++latest",
                "/O2",
                "/W4",
                "/EHsc",
                "/permissive-",
                "/wd4100",  # unreferenced parameter
                "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
                "/wd4189",  # local variable initialized but not referenced"
                "/wd4244",  # conversion from 'type1' to 'type2', possible loss of data
                "/wd4456",  # declaration of 'x' hides previous local declaration"
                "/wd4505",  # unreferenced local function has been removed"
                f"/I{include}",
                f"/I{include_dummy}",
                "/TP",
                *[str(h) for h in hpps],
            ]
        )

    if cpps:
        exe = snipdir / "a.exe"
        for p in [exe, *snipdir.glob("*.obj")]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        run(
            [
                cl,
                "/nologo",
                "/std:c++latest",
                "/O2",
                "/W4",
                "/EHsc",
                "/permissive-",
                "/wd4100",  # unreferenced parameter
                "/wd4146",  # unary minus operator applied to unsigned type, result still unsigned
                "/wd4189",  # local variable initialized but not referenced"
                "/wd4244",  # conversion from 'type1' to 'type2', possible loss of data
                "/wd4456",  # declaration of 'x' hides previous local declaration"
                "/wd4505",  # unreferenced local function has been removed"
                f"/I{include}",
                f"/I{include_dummy}",
                "/TP",
                *[str(c) for c in cpps],
                f"/Fe:{exe}",
            ]
        )
        run([str(exe)])


def compile_with_gcc_clang(snipdir: Path, include: Path, include_dummy: Path) -> None:
    cxxs: list[tuple[str, list[str]]] = []
    if which("g++"):
        cxxs.append(("g++", ["-Og", "-Wall", "-Wno-unused", "-Wextra", "-pedantic"]))
    if which("clang++"):
        cxxs.append(
            (
                "clang++",
                [
                    "-Og",
                    "-Wall",
                    "-Wno-unused",
                    "-Wextra",
                    "-pedantic",
                    "-Wno-unused-parameter",
                ],
            )
        )

    hpps = sorted(snipdir.glob("*.hpp"))
    cpps = sorted(snipdir.glob("*.cpp"))

    for cxx, base_flags in cxxs:
        for std in ("c++20", "c++23"):
            print_flush(f"{cxx.upper()}, {std.upper()}...")
            if hpps:
                run(
                    [
                        cxx,
                        f"--std={std}",
                        *base_flags,
                        "-fsyntax-only",
                        "-I",
                        str(include),
                        "-I",
                        str(include_dummy),
                        *[str(h) for h in hpps],
                    ]
                )

            if cpps:
                exe = snipdir / ("a.out" if os.name != "nt" else "a.exe")
                try:
                    exe.unlink()
                except FileNotFoundError:
                    pass
                run(
                    [
                        cxx,
                        f"--std={std}",
                        *base_flags,
                        "-I",
                        str(include),
                        "-I",
                        str(include_dummy),
                        *[str(c) for c in cpps],
                        "-o",
                        str(exe),
                    ]
                )
                run([str(exe)])


def main() -> None:
    os.chdir(repo_root())
    print_flush(f"Repo root: {os.getcwd()}")

    snipdir = Path(tempfile.mkdtemp())
    output_fd, _output_file = tempfile.mkstemp()
    output_file = Path(_output_file)
    print_flush(f"Created snippets directory: {snipdir}")

    def cleanup() -> None:
        print_flush(f"Deleting: {snipdir}, {output_file}")
        try:
            shutil.rmtree(snipdir)
        except FileNotFoundError:
            pass
        try:
            # Close the fd first
            os.close(output_fd)
            output_file.unlink()
        except FileNotFoundError:
            pass
        except PermissionError:
            print_flush(
                f"Warning: could not delete {output_file} due to permission error"
            )

    atexit.register(cleanup)

    include = repo_root() / "include"
    include_dummy = repo_root() / "docs_check" / "include"

    code, out = run_inner_check(snipdir, include)
    # Always write the output to a temp file to ease local inspection
    output_file.write_text(out, encoding="utf-8")
    if code != 0:
        sys.stdout.write(out)
        sys.stdout.flush()
        raise SystemExit(1)

    compare_expected(out)

    # Choose compilers
    have_any = any((which("cl"), which("g++"), which("clang++")))
    if not have_any:
        print_flush("No suitable compiler found (need cl, g++, or clang++)")
        raise SystemExit(1)

    print_flush()
    print_flush(f"Will compile: {snipdir}/")

    # Prefer MSVC on Windows if available, but also try GCC/Clang when present
    if which("cl"):
        compile_with_msvc(snipdir, include, include_dummy)

    # Also try GCC/Clang if available (mirrors bash script behavior)
    compile_with_gcc_clang(snipdir, include, include_dummy)

    print_flush("OK")


if __name__ == "__main__":
    main()
