"""Generate the Python code reference pages and navigation.

The `radex` Python package wraps compiled Cython extension modules, so the
package must be built and installed into the active environment before this
script runs (mkdocstrings introspects the imported modules at build time).
See docs/getting-started/installation.md.
"""

import importlib.util
from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src" / "python" / "src"
package_dir = src / "radex"

# `radex` ships as compiled Cython extension modules, so mkdocstrings can only
# document it once it has been built and installed into this environment. If
# it isn't available, skip generation and leave the hand-written
# docs/api/index.md page (which explains how to build it) as the landing page.
if importlib.util.find_spec("radex") is not None:
    for path in sorted(package_dir.rglob("*.py")) + sorted(package_dir.rglob("*.pyx")):
        if path.name == "__init__.py":
            continue

        # Skip internal/cache/build directories
        if "__pycache__" in path.parts:
            continue

        module_path = path.relative_to(src).with_suffix("")
        doc_path = path.relative_to(src).with_suffix(".md")
        full_doc_path = Path("api", doc_path)

        parts = tuple(module_path.parts)

        if parts[-1] == "__main__":
            continue

        # Strip the top-level 'radex' package name from the nav, but keep the
        # full dotted path for the mkdocstrings identifier.
        nav_parts = parts[1:] if parts[0] == "radex" else parts
        if nav_parts:
            nav[nav_parts] = doc_path.as_posix()

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"# {ident}\n\n::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
