# Russian Contract Redaction Toolkit

A fast hybrid redaction pipeline for Russian contracts combining Presidio (with custom recognizers), Natasha (NER), regex rules, and simple heuristics for role phrases and the "Предмет договора" section.

## Features

- Normalizes text (quotes, dashes, hyphenation across line breaks) with offset mapping back to original
- Presidio Analyzer + custom recognizers (INN/OGRN/KPP/BIK/RS/KS) with validation
- Natasha PERSON/ORG (with graceful degradation if heavy models are not present)
- Regex rules for ORG forms, role phrases ("именуемое в дальнейшем", "в лице … действующего на основании …")
- Heuristic detection of "Предмет договора" sections
- Conflict resolution and overlap merging with clear priorities
- Masking styles: tag, hash, random, asterisks; optional --keep-length
- Streaming-friendly processing and mirrored output directory
- JSONL log with spans metadata, salted deterministic hashes

## Installation

Create a virtual environment and install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional: Set a salt for deterministic hashing

```powershell
$env:REDACT_SALT = "your-salt-here"
```

Optional: Enable full Natasha NER (loads heavy models)

```powershell
$env:REDACTOR_NATASHA_FULL = "1"
```

## Usage

```powershell
python cli.py --input .\contracts --recursive --mask-style tag
python cli.py --input .\contract.txt --mask-style hash --keep-length
```

Options:
- --input: file or directory
- --output: output directory (default: ./redacted)
- --recursive: recurse into subdirectories
- --mask-style: tag|hash|random|asterisks
- --keep-length: preserve original length of masked fragments
- --min-score: minimum confidence score
- --workers: number of parallel workers
- --include SUBJECT: include subject detector (default)
- --dry-run: do not write files
- --disable-office: skip PDF/DOCX readers
- --log-level: INFO|DEBUG

## Output

- Redacted files are written into the mirrored structure under `./redacted/`
- A JSONL log `redacted/redaction_log.jsonl` is appended with entries:

```json
{
  "file": "path/to/file.txt",
  "start": 123,
  "end": 145,
  "label": "INN",
  "source": "Presidio Custom",
  "before_hash": "e3b0c442...",
  "after": "[[ENTITY:INN]]"
}
```

Original text is never written to logs.

## Notes & Limitations

- Presidio/Natasha are optional; the pipeline will degrade to what's available. For best quality, install all requirements.
- Natasha's full NER (ORG/PER) requires news embeddings; if not available, only NamesExtractor (PERSON) is used.
- Subject detection uses heuristics; verify outputs for edge cases. Max subject chunk is limited to 2000 chars.
- `--keep-length` will pad/truncate masks to match the original length; for tag/hash/random styles the info may be partially truncated.

## Testing

```powershell
pytest -q
```

The test suite includes validators for INN/OGRN and pattern detection for role phrases, plus basic pipeline checks.
