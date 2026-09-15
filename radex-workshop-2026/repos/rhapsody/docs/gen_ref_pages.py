"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Adjusted to your package structure
root = Path(__file__).parent.parent
src = root / "src"
package_dir = src / "rhapsody"

for path in sorted(package_dir.rglob("*.py")):
    if path.name == "__init__.py":
        continue

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

    # CHANGE HERE: Strip 'rhapsody' from the parts for navigation
    # but keep the full path for doc_path
    nav_parts = parts[1:] if parts[0] == "rhapsody" else parts  # Remove first element
    if nav_parts:  # Only add if there are parts left
        nav[nav_parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"# {ident}\n\n::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
