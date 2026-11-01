"""AST-based third-party import inventory.

Scans every .py under the project root (excluding .venv/, .git/ and __pycache__) and
writes, for each file, the third-party modules it imports together with the symbols
used from them. Intended consumers:
  * `docs/technical_documentation.md` D2 library/API matrix (no hand-typed rows)
  * the V12 gate: every matrix row must be present in this JSON
  * the drift guard in `build_docs.py --check`

Stdlib modules are recorded but flagged, so the matrix can separate "chosen" from
"bundled". Nothing here is hand-written; regenerate with
`python scripts/inventory_imports.py`.
"""
import ast
import json
import sys
from pathlib import Path

import importlib.metadata

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts" / "import_inventory.json"

# Packages distributed with Python. Anything not on this list is treated as a
# third-party dependency that must appear in requirements.txt and pip freeze.
STDLIB = {
    "__future__", "ast", "base64", "builtins", "csv", "datetime", "enum",
    "functools", "hashlib", "importlib", "inspect",
    "io", "itertools", "json", "logging", "math", "operator", "os", "pathlib",
    "platform", "pprint", "random", "re", "shutil", "socket", "stat", "string",
    "subprocess", "sys", "tempfile", "textwrap", "time", "traceback", "types",
    "typing", "unittest", "urllib", "warnings", "weakref",
    # Submodules: a top-level import of these is stdlib too.
    "collections", "concurrent", "contextlib", "copy", "email", "fractions",
    "gettext", "glob", "gzip", "html", "http", "ipaddress", "mailbox",
    "multiprocessing", "numbers", "queue", "selectors", "shlex", "signal",
    "site", "smtplib", "socketserver", "sqlite3", "statistics", "struct",
    "tarfile", "telnetlib", "test", "threading", "timeit", "tokenize",
    "tracemalloc", "turtle", "turtledemo", "unicodedata", "uuid",
    "venv", "wave", "webbrowser", "winreg", "wsgiref", "xml", "xmlrpc", "zipfile",
    "zipimport", "zlib",
}


def _version(name):
    """Best-effort installed version, or None if the package is not importable."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        # Fall back to the module's own metadata, if any.
        try:
            return getattr(__import__(name), "__version__", None)
        except Exception:
            return None


def _is_stdlib(name):
    """A module is stdlib if its top-level dotted component is a known stdlib name."""
    top = name.split(".")[0]
    return top in STDLIB


def scan_file(path):
    """Return ([third-party imports], [stdlib imports]) as {module: [symbols]}."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return {"__syntax_error__": [str(exc)]}, {}

    third_party = {}
    stdlib = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                symbol = alias.asname or alias.name.split(".")[0]
                bucket = stdlib if top in STDLIB else third_party
                bucket.setdefault(alias.name, set()).add(symbol)
        elif isinstance(node, ast.ImportFrom):
            # `from . import x` is relative and carries no resolvable module name.
            if node.module is None or node.level > 0:
                continue
            top = node.module.split(".")[0]
            bucket = stdlib if top in STDLIB else third_party
            for alias in node.names:
                bucket.setdefault(node.module, set()).add(alias.asname or alias.name)

    def freeze(mapping):
        return {k: sorted(v) for k, v in mapping.items()}

    return freeze(third_party), freeze(stdlib)


def main():
    files = sorted(
        p for p in ROOT.rglob("*.py")
        if ".venv" not in p.parts and ".git" not in p.parts
        and "__pycache__" not in p.parts and "site-packages" not in p.parts
    )

    inventory = {}
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        third_party, stdlib = scan_file(path)
        inventory[rel] = {"imports": third_party, "stdlib_imports": stdlib}

    # Every module the inventory mentions - third-party and stdlib alike - is
    # enriched with its installed version so the classification is auditable.
    modules = {m for entry in inventory.values()
               for source in ("imports", "stdlib_imports") for m in entry[source]}
    versions = {}
    for module in sorted(modules):
        top = module.split(".")[0]
        versions[module] = {
            "installed_version": _version(top),
            "is_stdlib": _is_stdlib(module),
        }

    payload = {
        "root": str(ROOT),
        "files": inventory,
        "modules": versions,
        "counts": {
            "files_scanned": len(inventory),
            "third_party_modules": len([m for m, v in versions.items()
                                        if not v["is_stdlib"]]),
            "stdlib_modules": len([m for m, v in versions.items()
                                   if v["is_stdlib"]]),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("scanned %d files: %d third-party modules, %d stdlib modules" % (
        payload["counts"]["files_scanned"], payload["counts"]["third_party_modules"],
        payload["counts"]["stdlib_modules"]))
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
else:
    # Allow other scripts to reuse the scanner without running it.
    pass