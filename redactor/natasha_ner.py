from __future__ import annotations

"""Natasha-based NER wrappers for PERSON and ORG extraction with graceful fallbacks.
"""
from dataclasses import dataclass
from typing import List, Optional
import os

try:
    from natasha import Doc, NewsNERTagger, NewsEmbedding, NamesExtractor  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    Doc = None  # type: ignore
    NewsNERTagger = None  # type: ignore
    NewsEmbedding = None  # type: ignore
    NamesExtractor = None  # type: ignore


@dataclass(frozen=True)
class NatashaSpan:
    start: int
    end: int
    label: str
    score: float
    source: str = "Natasha"


class NatashaNER:
    def __init__(self) -> None:
        # Lazy initialization to avoid heavy model load at import/test time
        self._emb = None
        self._ner = None
        self._names = None

    def analyze(self, text: str) -> List[NatashaSpan]:
        spans: List[NatashaSpan] = []
        # Ensure models are loaded lazily (opt-in via env to avoid heavy downloads by default)
        use_full = os.getenv("REDACTOR_NATASHA_FULL", "0").lower() in {"1", "true", "yes"}
        if use_full and self._ner is None and NewsEmbedding is not None and NewsNERTagger is not None and Doc is not None:
            try:
                self._emb = self._emb or NewsEmbedding()
                self._ner = self._ner or NewsNERTagger(self._emb)
            except Exception:
                self._emb = None
                self._ner = None
        if self._names is None and NamesExtractor is not None:
            try:
                self._names = NamesExtractor()
            except Exception:
                self._names = None

        # Full NER if available
        if self._ner is not None and Doc is not None:
            try:
                doc = Doc(text)
                doc.segment(self._emb)
                doc.tag_ner(self._ner)
                for span in doc.spans:
                    if span.type in {"PER", "ORG"}:
                        label = "PERSON" if span.type == "PER" else "ORG"
                        score = float(getattr(span, "score", 0.8))
                        spans.append(NatashaSpan(span.start, span.stop, label, score))
            except Exception:
                pass
        # NamesExtractor for PERSON as fallback
        if self._names is not None:
            try:
                matches = self._names(text)
                for m in matches:
                    spans.append(NatashaSpan(m.start, m.stop, "PERSON", 0.75))
            except Exception:
                pass
        return spans
