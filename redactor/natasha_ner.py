from __future__ import annotations

"""
Natasha-based NER wrapper focused on PERSON and ORG extraction.

Features:
- Full NER (PER, ORG) via NewsNERTagger when REDACTOR_NATASHA_FULL is set.
- Fallback PERSON via NamesExtractor.
- Deduplication: fallback PERSON won't duplicate/overlap NER results.
- Structured result via @dataclass.
- Defensive error handling with configurable logging.

Env vars:
- REDACTOR_NATASHA_FULL: "1"/"true"/"yes" to enable full NER (default: off)
- REDACTOR_NATASHA_LOG_LEVEL: Python logging level name (e.g. "INFO", "DEBUG")
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging
import os

# ---------- Logging ----------
_LOG_LEVEL = os.getenv("REDACTOR_NATASHA_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, _LOG_LEVEL, logging.WARNING))
log = logging.getLogger(__name__)

# ---------- Optional Natasha imports ----------
try:
    from natasha import (
        Doc,
        MorphVocab,
        NewsNERTagger,
        NewsEmbedding,
        NamesExtractor,
        Segmenter,
    )  # type: ignore[import-not-found]
except Exception as e:  # optional dependency may be missing
    log.info("Natasha import failed or partially unavailable: %s", e)
    Doc = None  # type: ignore
    NewsNERTagger = None  # type: ignore
    NewsEmbedding = None  # type: ignore
    NamesExtractor = None  # type: ignore
    Segmenter = None  # type: ignore


# ---------- Result type ----------
@dataclass(frozen=True)
class NatashaSpan:
    start: int
    end: int
    label: str  # "PERSON" | "ORG"
    score: float
    source: str = "Natasha"


# ---------- Helpers ----------
def _iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Intersection-over-Union for 1D spans [start, end)."""
    (a0, a1), (b0, b1) = a, b
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def _overlaps(a: Tuple[int, int], b: Tuple[int, int], iou_thr: float = 0.5) -> bool:
    """Whether spans overlap significantly (IoU >= iou_thr) or one contains the other."""
    (a0, a1), (b0, b1) = a, b
    if (a0 <= b0 and a1 >= b1) or (b0 <= a0 and b1 >= a1):  # containment
        return True
    return _iou((a0, a1), (b0, b1)) >= iou_thr


# ---------- Main class ----------
class NatashaNER:
    def __init__(self) -> None:
        # Lazy initialization
        self._emb: Optional[object] = None
        self._ner: Optional[object] = None
        self._names: Optional[object] = None
        self._segmenter: Optional[object] = None

    def _ensure_full_models(self) -> None:
        """Load heavy models for full NER mode (idempotent)."""
        if any(x is None for x in (Doc, NewsNERTagger, NewsEmbedding, Segmenter)):
            log.debug("Full NER components unavailable in environment.")
            return
        if self._ner is None or self._emb is None or self._segmenter is None:
            try:
                self._emb = self._emb or NewsEmbedding()
                self._ner = self._ner or NewsNERTagger(self._emb)
                self._segmenter = self._segmenter or Segmenter()
                log.debug("Full NER models initialized.")
            except Exception as e:
                log.warning("Failed to init full NER models: %s", e)
                self._emb = None
                self._ner = None
                self._segmenter = None

    def _ensure_names(self) -> None:
        """Load lightweight NamesExtractor (idempotent)."""
        if NamesExtractor is None:
            return
        if self._names is None:
            try:
                morph = MorphVocab()
                self._names = NamesExtractor(morph)
                log.debug("NamesExtractor initialized.")
            except Exception as e:
                log.warning("Failed to init NamesExtractor: %s", e)
                self._names = None

    def analyze(self, text: str) -> List[NatashaSpan]:
        spans: List[NatashaSpan] = []

        use_full = os.getenv("REDACTOR_NATASHA_FULL", "0").lower() in {"1", "true", "yes"}
        if use_full:
            self._ensure_full_models()
        self._ensure_names()

        # -------- Full NER path (PER + ORG) --------
        if use_full and self._ner is not None and Doc is not None and self._segmenter is not None:
            try:
                doc = Doc(text)  # type: ignore[operator]
                doc.segment(self._segmenter)  # correct: needs Segmenter, not embeddings
                doc.tag_ner(self._ner)
                for span in doc.spans:
                    if span.type in {"PER", "ORG"}:
                        label = "PERSON" if span.type == "PER" else "ORG"
                        score = float(getattr(span, "score", 0.8))
                        spans.append(NatashaSpan(span.start, span.stop, label, score))
                log.debug("Full NER produced %d spans.", len(spans))
            except Exception as e:
                log.warning("Full NER failed: %s", e)

        # -------- Fallback: NamesExtractor (PERSON only) --------
        # Deduplicate against already found spans (prefer full NER results)
        if self._names is not None:
            try:
                existing = [(s.start, s.end) for s in spans if s.label == "PERSON"]
                matches = self._names(text)
                added = 0
                for m in matches:
                    s = getattr(m, "start", None)
                    e = getattr(m, "stop", None)
                    if s is None or e is None:
                        span = getattr(m, "span", None)
                        if span and isinstance(span, (tuple, list)) and len(span) == 2:
                            s, e = span
                        else:
                            continue
                    # skip if this PERSON overlaps an existing PERSON from full NER
                    if any(_overlaps((s, e), ex) for ex in existing):
                        continue
                    spans.append(NatashaSpan(s, e, "PERSON", 0.75))
                    added += 1
                log.debug("Fallback NamesExtractor added %d PERSON spans.", added)
            except Exception as e:
                log.warning("NamesExtractor failed: %s", e)

        return spans
# ---------- End of natasha_ner.py ----------