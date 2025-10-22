from __future__ import annotations

"""Main redaction pipeline: normalization, detection, aggregation, masking, and I/O orchestration."""
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import os
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .patterns import regex_find_spans, LABEL_SUBJECT
from .presidio_recognizers import build_analyzer, analyze_text
from .natasha_ner import NatashaNER


class MaskStyle(str, Enum):
    TAG = "tag"
    HASH = "hash"
    RANDOM = "random"
    ASTERISKS = "asterisks"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float
    source: str
    labels_joined: Optional[str] = None  # for logging multiple labels


@dataclass
class PipelineOptions:
    mask_style: MaskStyle = MaskStyle.TAG
    keep_length: bool = False
    min_score: float = 0.5
    include_subject: bool = True
    salt_env: str = "REDACT_SALT"


SOURCE_PRIORITY = {
    "Regex": 3,
    "Presidio Custom": 2,
    "Natasha": 1,
    "Presidio Default": 0,
}


class Pipeline:
    def __init__(self, options: Optional[PipelineOptions] = None) -> None:
        self.opts = options or PipelineOptions()
        self.analyzer = build_analyzer()
        self.natasha = NatashaNER()
        self.rng = random.Random(12345)
        self.log = logging.getLogger("redactor.pipeline")

    # --- Normalization with offset mapping ---
    @staticmethod
    def normalize_with_mapping(text: str) -> Tuple[str, List[int]]:
        """Normalize quotes/dashes/hyphenation and return mapping from normalized indices to original.

        - Replace «» with ""
        - Normalize different dashes to '-'
        - Join hyphenation across line breaks: "компа-\nния" -> "компания"
        - Convert \r\n to \n
        Returns (normalized_text, norm_to_orig_index_map)
        """
        orig = text.replace("\r\n", "\n")
        norm_chars: List[str] = []
        map_idx: List[int] = []

        i = 0
        L = len(orig)
        while i < L:
            ch = orig[i]
            # quotes normalization (same length)
            if ch in "«»”„":
                norm_chars.append('"')
                map_idx.append(i)
                i += 1
                continue
            # dash variants
            if ch in "—–−":
                norm_chars.append('-')
                map_idx.append(i)
                i += 1
                continue
            # hyphenation join: letter '-' '\n' letter
            if (
                ch.isalpha()
                and i + 2 < L
                and orig[i + 1] == '-'
                and orig[i + 2] == '\n'
                and i + 3 < L
                and orig[i + 3].isalpha()
            ):
                # keep current letter
                norm_chars.append(ch)
                # map_idx.append(i)
                map_idx.extend([i, i+1, i+2])
                # skip '-' and '\n'
                i += 1  # now at '-'
                i += 1  # now at '\n'
                i += 1  # now at next letter
                continue
            # default: copy
            norm_chars.append(ch)
            map_idx.append(i)
            i += 1

        return ("".join(norm_chars), map_idx)

    # --- Detection ---
    def detect_spans(self, norm_text: str) -> List[Span]:
        spans: List[Span] = []

        # Presidio
        pres = analyze_text(self.analyzer, norm_text, min_score=self.opts.min_score)
        for d in pres:
            spans.append(Span(d["start"], d["end"], d["label"], d["score"], d["source"]))

        # Natasha
        for s in self.natasha.analyze(norm_text):
            spans.append(Span(s.start, s.end, s.label, s.score, s.source))

        # Regex
        for s in regex_find_spans(norm_text):
            # Optionally filter out subject spans if disabled
            if s.label == LABEL_SUBJECT and not self.opts.include_subject:
                continue
            spans.append(Span(s.start, s.end, s.label, s.score, s.source))

        return spans

    # --- Aggregation & conflict resolution ---
    def _choose_primary(self, a: Span, b: Span) -> Span:
        # Higher score first
        if abs(a.score - b.score) > 1e-9:
            return a if a.score > b.score else b
        # Tie-breaker: source priority
        pa = SOURCE_PRIORITY.get(a.source, 0)
        pb = SOURCE_PRIORITY.get(b.source, 0)
        if pa != pb:
            return a if pa > pb else b
        # Otherwise, prefer longer span (more specific for ROLE/SUBJECT blocks)
        la = a.end - a.start
        lb = b.end - b.start
        return a if la >= lb else b

    @staticmethod
    def _overlap(a: Span, b: Span) -> bool:
        return not (a.end <= b.start or b.end <= a.start)

    def merge_overlaps(self, spans: List[Span]) -> List[Span]:
        if not spans:
            return []
        spans = sorted(spans, key=lambda s: (s.start, -s.end))
        merged: List[Span] = []
        current = spans[0]
        for s in spans[1:]:
            if self._overlap(current, s):
                # choose primary
                primary = self._choose_primary(current, s)
                other = s if primary is current else current
                # If ROLE and PERSON overlap, union and keep ROLE as label
                labels_joined = None
                if {primary.label, other.label} == {"ROLE", "PERSON"}:
                    labels_joined = "ROLE+PERSON"
                # merge bounds to single span
                new_start = min(current.start, s.start)
                new_end = max(current.end, s.end)
                current = Span(new_start, new_end, primary.label, primary.score, primary.source, labels_joined)
            else:
                merged.append(current)
                current = s
        merged.append(current)
        return merged

    # --- Mapping back to original offsets ---
    @staticmethod
    def map_spans_to_original(spans: List[Span], norm_to_orig: List[int]) -> List[Span]:
        mapped: List[Span] = []
        L = len(norm_to_orig)
        for s in spans:
            ns = max(0, min(s.start, L - 1))
            ne = max(0, min(s.end - 1, L - 1))
            start_orig = norm_to_orig[ns]
            end_orig = norm_to_orig[ne] + 1  # exclusive
            mapped.append(Span(start_orig, end_orig, s.label, s.score, s.source, s.labels_joined))
        return mapped

    # --- Mask generation ---
    def _hash_mask(self, text: str) -> str:
        salt = os.getenv(self.opts.salt_env, "")
        h = hashlib.sha256((salt + text).encode("utf-8", errors="ignore")).hexdigest()
        return h[:12]

    def _random_mask(self, length: int) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(self.rng.choice(alphabet) for _ in range(max(6, min(24, length))))

    def _tag_mask(self, label: str) -> str:
        return f"[[ENTITY:{label}]]"

    def _asterisks_mask(self, length: int) -> str:
        return "*" * max(1, length)

    def _mask_for(self, label: str, original: str, keep_length: bool) -> str:
        if self.opts.mask_style == MaskStyle.TAG:
            mask = self._tag_mask(label)
        elif self.opts.mask_style == MaskStyle.HASH:
            mask = self._hash_mask(original)
        elif self.opts.mask_style == MaskStyle.RANDOM:
            mask = self._random_mask(len(original))
        else:
            mask = self._asterisks_mask(len(original))

        if not keep_length:
            return mask
        # adjust to same length
        target = len(original)
        if len(mask) == target:
            return mask
        if len(mask) > target:
            return mask[:target]
        # pad with '*'
        return (mask + ("*" * (target - len(mask))))

    # --- Apply masking ---
    def apply_masks(self, text: str, spans: List[Span]) -> Tuple[str, List[dict]]:
        if not spans:
            return text, []
        spans = sorted(spans, key=lambda s: (s.start, s.end))
        out = []
        logs: List[dict] = []
        i = 0
        for s in spans:
            if s.start < i:
                continue  # skip invalid ordering
            out.append(text[i:s.start])
            fragment = text[s.start:s.end]
            mask = self._mask_for(s.label, fragment, self.opts.keep_length)
            out.append(mask)
            before_hash = hashlib.sha256(fragment.encode("utf-8", errors="ignore")).hexdigest()
            logs.append({
                "start": s.start,
                "end": s.end,
                "label": s.labels_joined or s.label,
                "source": s.source,
                "before_hash": before_hash,
                "after": mask,
            })
            i = s.end
        out.append(text[i:])
        return ("".join(out), logs)

    # --- End-to-end for one text ---
    def process_text(self, text: str) -> Tuple[str, List[dict]]:
        norm_text, norm_map = self.normalize_with_mapping(text)
        spans = self.detect_spans(norm_text)
        spans = self.merge_overlaps(spans)
        spans = self.map_spans_to_original(spans, norm_map)
        spans = self.merge_overlaps(spans)  # re-merge after mapping to be safe
        redacted, logs = self.apply_masks(text, spans)
        return redacted, logs
