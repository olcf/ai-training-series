"""Generate the Python code reference pages and navigation.

Documents the `rhapsody_plugins.openfoam` package under
`src/python/rhapsody_plugins`. The package (and its `dragon`/`rhapsody`
dependencies) must be importable in the documentation build environment for
mkdocstrings to introspect it. See docs/getting-started/installation.md.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src" / "python"
package_dir = src / "rhapsody_plugins"

for path in sorted(package_dir.rglob("*.py")):
    # Skip internal/cache directories
    if "__pycache__" in path.parts:
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    if not parts:
        continue

    # Strip the top-level 'rhapsody_plugins' package name from the nav, but
    # keep the full dotted path for the mkdocstrings identifier.
    nav_parts = parts[1:] if parts[0] == "rhapsody_plugins" else parts
    if nav_parts:
        nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
