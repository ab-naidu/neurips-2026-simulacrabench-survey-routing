"""Package a candidate from its runtime files with reproducible archive bytes."""

import argparse
import hashlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if not (args.source / "main.py").is_file():
        parser.error("candidate directory must contain main.py")
    files = [p for p in args.source.iterdir() if p.is_file()
             and (p.suffix == ".py" or p.name in {"requirements.txt", "models.txt", "LICENSE"})]
    with ZipFile(args.output, "w", compression=ZIP_STORED) as archive:
        for path in sorted(files, key=lambda p: p.name):
            entry = ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            # Normalize text checkouts across Windows and Linux.
            content = path.read_bytes().replace(b"\r\n", b"\n")
            archive.writestr(entry, content)
    print(f"{hashlib.sha256(args.output.read_bytes()).hexdigest()}  {args.output.name}")


if __name__ == "__main__":
    main()
