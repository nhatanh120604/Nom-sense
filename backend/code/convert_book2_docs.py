"""Batch convert Word documents in the Book2 folder to PDF using Microsoft Word."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

try:
    from docx2pdf import convert
except ImportError as exc:  # pragma: no cover - dependency may be missing locally
    raise SystemExit(
        "The 'docx2pdf' package is required. Install it with 'pip install docx2pdf'."
    ) from exc


def iter_documents(folder: Path) -> Iterable[Path]:
    for pattern in ("*.doc", "*.docx"):
        yield from folder.glob(pattern)


def convert_book2_documents(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in iter_documents(input_dir):
        target = output_dir / f"{path.stem}.pdf"
        convert(str(path), str(target))
        print(f"Converted {path.name} -> {target.name}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "Word" / "Book2"
    default_output = default_input / "pdf"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Folder containing .doc or .docx files (default: {default_input})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Folder to write PDFs to (default: {default_output})",
    )
    args = parser.parse_args()

    convert_book2_documents(args.input, args.output)


if __name__ == "__main__":
    main()
