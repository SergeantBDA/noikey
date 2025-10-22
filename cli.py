from __future__ import annotations

"""CLI interface for the redaction toolkit."""
import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from redactor.pipeline import Pipeline, PipelineOptions, MaskStyle
from redactor.io_utils import discover_files, read_any, mirror_output_path, write_text, append_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mass redaction of Russian contracts")
    p.add_argument("--input", required=True, help="Path to input file or directory")
    p.add_argument("--output", default="redacted", help="Output directory (mirrored structure)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    p.add_argument("--mask-style", choices=[m.value for m in MaskStyle], default=MaskStyle.TAG.value)
    p.add_argument("--keep-length", action="store_true", help="Preserve original length of masked fragments")
    p.add_argument("--min-score", type=float, default=0.5, help="Minimum score threshold")
    p.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    p.add_argument("--include", choices=["SUBJECT"], nargs="*", default=["SUBJECT"], help="Optional detectors to include")
    p.add_argument("--dry-run", action="store_true", help="Do not write files, only report")
    p.add_argument("--log-level", default="INFO", choices=["INFO", "DEBUG"], help="Log level")
    p.add_argument("--disable-office", action="store_true", help="Disable reading .pdf/.docx (txt only)")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def process_one(pipeline_opts: PipelineOptions, input_root: Path, output_root: Path, path: Path, dry: bool, enable_office: bool) -> dict:
    pipe = Pipeline(pipeline_opts)
    text = read_any(path, enable_office=enable_office)
    if text is None:
        return {"file": str(path), "status": "skipped", "reason": "unreadable or disabled"}
    redacted, logs = pipe.process_text(text)
    if not dry:
        out_path = mirror_output_path(path, input_root.resolve(), output_root.resolve())
        log_path = output_root / "redaction_log.jsonl"
        write_text(out_path, redacted)
        for entry in logs:
            entry_with_file = dict(entry)
            entry_with_file["file"] = str(path)
            append_jsonl(log_path, entry_with_file)
    # Summary by label
    summary = {}
    for e in logs:
        lab = e["label"]
        summary[lab] = summary.get(lab, 0) + 1
    total_chars = sum(e["end"] - e["start"] for e in logs)
    return {"file": str(path), "status": "ok", "replacements": summary, "chars_masked": total_chars}


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    opts = PipelineOptions(
        mask_style=MaskStyle(args.mask_style),
        keep_length=bool(args.keep_length),
        min_score=float(args.min_score),
        include_subject=("SUBJECT" in (args.include or [])),
    )

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    files = discover_files(input_root, recursive=bool(args.recursive))
    if not files:
        logging.warning("No input files found: %s", input_root)
        return

    enable_office = not bool(args.disable_office)

    results: List[dict] = []
    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(process_one, opts, input_root, output_root, p, bool(args.dry_run), enable_office)
                for p in files
            ]
            for f in as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    logging.exception("Worker failed: %s", e)
    else:
        for p in files:
            try:
                results.append(process_one(opts, input_root, output_root, p, bool(args.dry_run), enable_office))
            except Exception as e:
                logging.exception("Processing failed for %s: %s", p, e)

    # Print summary
    total_files = len(results)
    ok = sum(1 for r in results if r.get("status") == "ok")
    masked = sum(r.get("chars_masked", 0) for r in results if r.get("status") == "ok")
    logging.info("Processed %d files, ok=%d, chars_masked=%d", total_files, ok, masked)
    # Per-type aggregation
    agg: dict = {}
    for r in results:
        if r.get("status") != "ok":
            continue
        for k, v in (r.get("replacements") or {}).items():
            agg[k] = agg.get(k, 0) + int(v)
    logging.info("Replacements by type: %s", json.dumps(agg, ensure_ascii=False))


if __name__ == "__main__":
    main()
