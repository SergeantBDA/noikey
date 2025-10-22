from __future__ import annotations

"""I/O utilities: reading text from files and writing outputs + JSONL logs."""
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_docx_file(path: Path) -> Optional[str]:  # optional dependency
    try:
        import docx  # type: ignore
    except Exception:
        return None
    try:
        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return None


def read_pdf_file(path: Path) -> Optional[str]:  # optional dependency
    try:
        from pdfminer.high_level import extract_text  # type: ignore
    except Exception:
        return None
    try:
        return extract_text(str(path))
    except Exception:
        return None


def discover_files(root: Path, recursive: bool) -> List[Path]:
    files: List[Path] = []
    if root.is_file():
        return [root]
    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        for name in filenames:
            p = dp / name
            if p.suffix.lower() in {".txt", ".pdf", ".docx"}:
                files.append(p)
        if not recursive:
            break
    return files


def read_any(path: Path, enable_office: bool = True) -> Optional[str]:
    suf = path.suffix.lower()
    if suf == ".txt":
        return read_text_file(path)
    if not enable_office:
        return None
    if suf == ".docx":
        return read_docx_file(path)
    if suf == ".pdf":
        return read_pdf_file(path)
    return None


def mirror_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root) if input_path.is_absolute() else input_path
    out = output_root / rel
    return out.with_suffix(".txt")


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
