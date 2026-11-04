"""Verify the distributed snapshot with Python's standard library."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    failures = []
    count = 0
    for line in (ROOT / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(None, 1)
        path = (ROOT / name.lstrip("* ")).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            failures.append(f"Missing or invalid path: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"Checksum mismatch: {name}")
        count += 1
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {count} files match CHECKSUMS.sha256")


if __name__ == "__main__":
    main()
