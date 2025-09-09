# Documentation tests

Execute `check` (shell script without suffix) to check the documentation for broken {links/snippets/style}.
Arguments and cwd are ignored, execute in any directory.

On Windows, use `python docs_check/check.py` from a Developer Command Prompt (so `cl.exe` is available) or with other compilers (MinGW `g++`, LLVM `clang++`) on PATH. The Python runner mirrors the shell script and supports MSVC, GCC and Clang.


## Link and style checks

The following properties of the Markdown files are checked:

- links must lead to existing markdown files and/or existing anchors (`Foo.md` or `Foo.md#bar` or `#bar`)
- exactly one H1 (`#`) per file, on the first line, matching filename
- each H2 (`##`) preceded by exactly two blank lines
- H3, H4, H5 (`###[#[#]]`) preceded by exactly one blank line
- all headings followed by exactly one blank line

Exceptions can be added to `expected-warnings` - the order is important.
The result of the check is compared to `expected-warnings` using `git diff`.
Red lines ("removals") mean errors in Markdown files missing from `expected-warnings`.
Green lines ("additions") mean `expected-warnings` entries that did not happen.

When running in CI, all unexpected warnings are fatal (script does stops and returns failure status).
Otherwise, the list of warning does not cause the script to stop nor return failure status.


## Snippet checks

Each Markdown file is converted to up to two C++ files.
One file for block-level declarations and statements.
One file for namespace-level declarations (used for templates).
The files are compiled separately, neither is included in the other.

The block-level file (used for non-templates) looks roughly like this:

```cpp
...()
{
  snippet 1;
  {
    snippet 2;
    {
      snippet 3;
    }
  }
}
```

The namespace-level file (used for templates) looks roughly like this:

```cpp
namespace ... {
  snippet 1;
  namespace ... {
    snippet 2;
    namespace ... {
      snippet 3;
    }
  }
}
```

This allows us to use names from previous snippets without collisions when the names are redefined.

Snippets verification can be customized in `config.py`:
- `global_decls` is a multiline string containing definitions visible for all snippets
  - all noarr `.hpp` files are added implicitly
- `substitutions` defines per-snippet substitutions
  - each snippet is identified by the file name and index of snippet within that file
    (this is more stable than line numbers, albeit not perfect)
  - the substitutions are given in the format `{'pattern': 'replacement', ...}`,
    where `pattern` is a regex (use `PROLOG` or `EPILOG` without quotes to match snippet start/end)
  - `UNIQ` is a macro that expands to a different identifier every time,
    can be used to avoid colliding identifiers within the same snippet
- the `convert_synopsis` function is used to preprocess all snippets of type `hpp`

If you get C++ errors and are unsure what even is in the generated C++ files,
you can pause the checks and inspect the files. To do that:
- run the `check` program in terminal
- wait until you see `Will compile: /tmp/tmp.XXXXXXXXXX/` (for some `X`s)
- press `^Z` (control-Z) to pause the script and get shell
- the files are in the mentioned `/tmp/tmp.XXXXXXXXXX/` directory
- once you are done, do `fg` to resume (optionally followed by `^C` to cancel), so the files get deleted
