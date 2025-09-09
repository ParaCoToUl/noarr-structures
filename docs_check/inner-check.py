if __name__ != "__main__":
    raise Exception("do not import me")

import sys, os, os.path
import re
import json

from config import global_decls, substitutions, convert_synopsis
from pathlib import Path
from typing import Callable, List, TypeVar

Pattern = re.compile


###############
# Definitions #
###############

T = TypeVar("T")


def scope_block(f: Callable[[], T]) -> T:
    return f()


SNIP_PATT = Pattern("\n```([a-z]+)\n(.*?\n)```\n", re.DOTALL)
HEAD_PATT = Pattern("(\n+)(#+) ([^\n]+)(?=(\n+))", re.DOTALL)
TICK_PATT = Pattern("`.*?`")
LINK_PATT = Pattern(r"\[(.*?)\]\((.*?)\)")
CUDA_PATT = Pattern("(.*)<<<(.*)>>>(.*)")

ANCHOR_IGN_PATT = Pattern("[(),.:<>`]")


def clean_anchor(text: str) -> str:
    return ANCHOR_IGN_PATT.sub("", text).replace(" ", "-").lower()


def clean_anchor_1(text: str) -> str:
    return ANCHOR_IGN_PATT.sub("", text).replace(" ", "")


def cuda_replacement(mo: re.Match[str]) -> str:
    kernel, params, rest = mo.group(1, 2, 3)
    return f"void UNIQ() {{ snippet_check_kernel_params({params}); {kernel}{rest} }}"


def filepath_to_name(file_path: str) -> str:
    return file_path.replace("/", "%").replace("\\", "%").replace(":", "%")


def canonical_filename(fn: str) -> str:
    return fn.replace("\\", "/")


##################
# Initialization #
##################

cmd = sys.argv[0]
try:
    [cmd, _snipdir, include, *args] = sys.argv
    snipdir = Path(_snipdir)
    if not args:
        raise Exception
except:
    print("Usage:", cmd, "<snippets output dir> <include dir> <markdown files ...>")
    sys.exit(1)

if os.listdir(snipdir):
    print(snipdir, "(snippet output dir) is not empty")
    sys.exit(1)

available: set[str] = set()
linked: list[tuple[str, str, bool]] = []
snipfns = 0


@scope_block
def _():
    inc = include
    if not include.endswith("/"):
        inc += "/"

    def rel_include(dn: str) -> str:
        if not dn.startswith(inc):
            raise Exception(dn, "does not start with", inc)
        return dn[len(inc) :]

    with open(f"{snipdir}/snippet_check_all.hpp", "x") as f:
        headers = sorted(
            f"{rel_include(dn)}/{fn}" for dn, _, fns in os.walk(inc) for fn in fns
        )
        if not headers:
            print("Include directory empty or missing")
            exit(1)
        for h in headers:
            if not h.endswith(".hpp"):
                continue
            elif h.endswith("/omp.hpp") or h.endswith("/std_para.hpp"):
                continue
            print(f"#include <{h}>", file=f)
        print(global_decls, file=f)
        print("#define UNIQ_A(CTR) UNIQ_##CTR", file=f)
        print("#define UNIQ_B(CTR) UNIQ_A(CTR)", file=f)
        print("#define UNIQ UNIQ_B(__COUNTER__)", file=f)
        print("extern struct snippet_check_ret {} snippet_check_ret;", file=f)
        print("static void snippet_check_kernel_params(dim3, dim3) {}", file=f)


########
# Main #
########

for file in args:
    with open(file, newline="") as f:
        content = f.read()

    if sys.platform == "win32":
        content = content.replace("\r\n", "\n")

    canonical_file = canonical_filename(file)

    ##########################
    # Basic file type checks #
    ##########################
    if not content:
        print(canonical_file, "is empty")
        continue
    if "\ufeff" in content:
        print(canonical_file, "contains BOM")
        continue
    if "\r" in content:
        print(canonical_file, "contains CR")
    if content[-1] != "\n":
        print(canonical_file, "last line misses LF")

    #######################################
    # Separate the snippets from the text #
    #######################################
    snippets: List[tuple[int, int, str]] = (
        []
    )  # tuples of (snippet index in file, byte offset in file, c++ source)
    template_snippets: List[tuple[int, int, str]] = (
        []
    )  # same format, but they go in a namespace scope instead of fn scope

    def replacement(mo: re.Match[str]) -> str:
        idx = len(snippets) + len(template_snippets)
        code = mo.group(2)
        match mo.group(1):
            case "cpp":
                coll = template_snippets if "template<" in code else snippets
            case "cu":
                code = CUDA_PATT.sub(cuda_replacement, code.replace("__global__", ""))
                coll = template_snippets if "template<" in code else snippets
            case "hpp" | "cuh":
                code = convert_synopsis(code)
                if not code:
                    print(canonical_file, "contains unparsable synopsis")
                coll = template_snippets
            case _:
                print(canonical_file, "contains unrecognized snippet type", mo.group(1))
                coll = []  # append to nowhere
        coll.append((idx, mo.start(2), code))
        return "\n<!-- BLOCK SNIPPET -->\n"

    text = SNIP_PATT.sub(replacement, content)

    ######################################
    # Write out snippets for compilation #
    ######################################
    if snippets:

        @scope_block
        def _():
            with open(snipdir / f"{filepath_to_name(file)}.cpp", "x") as f:
                print('#include "snippet_check_all.hpp"', file=f)
                print(file=f)
                print(f"struct snippet_check_ret &snippet_check_fn_{snipfns}()", file=f)
                for idx, off, code in snippets:
                    for p, r in substitutions.get((canonical_file, idx), {}).items():
                        code = re.sub(p, r, code, flags=re.MULTILINE)
                    line = content[:off].count("\n") + 1
                    print("{", file=f)
                    print(f"#line {line} {json.dumps(file)} // {idx}", file=f)
                    f.write(code)
                    print(f'#line {1000*idx+1} "GENERATED"', file=f)
                    print("", file=f)
                print(f"return snippet_check_ret;", file=f)
                for _ in snippets:
                    print("}", file=f)

        snipfns += 1
    if template_snippets:

        @scope_block
        def _():
            with open(snipdir / f"{filepath_to_name(file)}.tmpl.cpp", "x") as f:
                print('#include "snippet_check_all.hpp"', file=f)
                print(file=f)
                print("namespace {", file=f)
                for idx, off, code in template_snippets:
                    for p, r in substitutions.get((canonical_file, idx), {}).items():
                        code = re.sub(p, r, code, flags=re.MULTILINE)
                    line = content[:off].count("\n") + 1
                    print("namespace UNIQ {", file=f)
                    print(f"#line {line} {json.dumps(file)} // {idx}", file=f)
                    f.write(code)
                    print(f'#line {1000*idx+1} "GENERATED"', file=f)
                    print("", file=f)
                for _ in template_snippets:
                    print("}", file=f)
                print("} // unnamed namespace", file=f)

    #############################################################
    # There should be exactly one H1, located on the first line #
    #############################################################
    if not text.startswith("# "):
        print(canonical_file, "does not start with a H1")
        continue
    if "\n# " in text:
        print(canonical_file, "contains stray H1")

    #################################
    # The H1 should match file name #
    #################################
    @scope_block
    def _():
        h1len = text.index("\n")
        if text[h1len:][:2] != "\n\n":
            print(canonical_file, "H1 not followed by empty line")
        if text[h1len + 2] == "\n":
            print(canonical_file, "H1 followed by too many empty lines")
        if clean_anchor_1(text[2:h1len]) + ".md" != file.split(os.path.sep)[-1]:
            print(canonical_file, "H1 does not match filename")

    ##########################
    # H1 only without anchor #
    ##########################
    available.add(file)

    ####################################
    # Check and collect other headings #
    ####################################
    @scope_block
    def _():
        headings = HEAD_PATT.findall(text)
        if len(headings) != text.count("\n#"):
            print(canonical_file, "contains unmatched heading (bug in regex?)")
        for nl_before, hashes, htext, nl_after in headings:
            level = len(hashes)
            if level < 2 or level > 5:
                print(canonical_file, "contains suspicious heading: level", level)
            if len(nl_after) != 2:
                print(canonical_file, "contains bad num of blank lines after", htext)
            if len(nl_before) != (3 if level == 2 else 2):
                print(canonical_file, "contains bad num of blank lines before", htext)
            available.add(file + "#" + clean_anchor(htext))

    ###########################
    # Check other blank lines #
    ###########################
    if "\n\n\n" in text.replace("\n\n\n#", ""):
        print(
            canonical_file, "contains doubled blank lines (not followed by a heading)"
        )

    #############################################################
    # Remove inline snippets so they don't interfere with links #
    #############################################################
    text = TICK_PATT.sub("<!-- SNIPPET -->", text)
    if "`" in text:
        print(canonical_file, "contains stray backtick or unmatched snippet")
        continue

    ##########################################################
    # Remember links (check later, after collecting targets) #
    ##########################################################
    @scope_block
    def _():
        for link in LINK_PATT.finditer(text):
            (full, lntext, target) = link.group(0, 1, 2)
            if (
                lntext.count("(") != lntext.count(")")
                or "[" in lntext
                or "]" in lntext
                or "(" in target
                or ")" in target
                or "[" in target
                or "]" in target
            ):
                print(canonical_file, "contains corrupted link", full)
            is_img = text[link.start() - 1] == "!"
            linked.append((file, target, is_img))


###############
# Check links #
###############
@scope_block
def _():
    for source, target, is_img in linked:
        if target.startswith("https://"):
            continue

        if target.startswith("#"):
            if is_img:
                print("Invalid image linking to anchor")
                continue
            resolved = source + target
        else:
            left = source.split(os.path.sep)
            left.pop()
            right = target
            while right.startswith("../"):
                left.pop()
                right = right[3:]
            left.append(right.replace("/", os.path.sep))
            resolved = os.path.sep.join(left)
            del left, right

        if is_img:  # file
            if not os.path.isfile(resolved):
                print(
                    "Unresolved image",
                    dict(source=source, relative=target, absolute=resolved),
                )
        else:  # markdown link
            if not resolved in available:
                print(
                    "Unresolved",
                    dict(
                        source=canonical_filename(source),
                        relative=target,
                        absolute=canonical_filename(resolved),
                    ),
                )


###################
# Main executable #
###################
@scope_block
def _():
    with open(snipdir / "main.cpp", "x") as f:
        print("#include <cstdlib>", file=f)
        print("struct snippet_check_ret {} snippet_check_ret;", file=f)
        print("int main() {", file=f)
        for i in range(snipfns):
            print(f"extern struct snippet_check_ret &snippet_check_fn_{i}();", file=f)
            print(
                f"if(&snippet_check_fn_{i}() != &snippet_check_ret) std::abort();",
                file=f,
            )
        print("return 0;", file=f)
        print("}", file=f)
