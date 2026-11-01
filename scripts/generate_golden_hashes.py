"""Build the golden sha256 manifest for the committed artifacts.

Run from the project root after any legitimate change to a produced artifact:

    python scripts/generate_golden_hashes.py

The manifest is consumed by the Tier T2 determinism tests, so regenerating it is the
explicit step that says "these new outputs are the intended baseline".
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ARTIFACTS / "golden_hashes.json"

EXTENSIONS = {".csv", ".json", ".joblib"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest = {}
    for path in sorted(ARTIFACTS.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            manifest[path.name] = sha256_of(path)
    OUT.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} hashes to {OUT}")
    for name, digest in sorted(manifest.items()):
        print(f"  {name:42s} {digest[:16]}...")


if __name__ == "__main__":
    main()