from __future__ import annotations

"""
Natasha-based NER wrapper focused on PERSON and ORG extraction.

Features:
- Full NER (PER, ORG) via NewsNERTagger when REDACTOR_NATASHA_FULL is set.
- Fallback PERSON via NamesExtractor with lemma-based filtering.
- Deduplication: fallback PERSON won't duplicate/overlap NER results.
- Structured result via @dataclass.
- Defensive error handling with configurable logging.

Env vars:
- REDACTOR_NATASHA_FULL: "1"/"true"/"yes" to enable full NER (default: off)
- REDACTOR_NATASHA_LOG_LEVEL: Python logging level name (e.g. "INFO", "DEBUG")
"""
import os
import re
import logging
from dataclasses import dataclass
from typing      import List, Optional, Tuple
from functools   import lru_cache
from pathlib     import Path

# ---------- Logging ----------
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
        AddrExtractor,
    )
except Exception as e:
    log.info("Natasha import failed or partially unavailable: %s", e)
    Doc = None
    NewsNERTagger = None
    NewsEmbedding = None
    NamesExtractor = None
    Segmenter = None

# ---------- Morph + stopwords ----------
try:
    import pymorphy2
    _morph = pymorphy2.MorphAnalyzer()
except Exception:
    _morph = None

try:
    from stop_words import get_stop_words
except Exception:
    get_stop_words = None

try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
except Exception:
    nltk_stopwords = None

from razdel import tokenize

ADDR_HINTS = r"(ул|улица|шоссе|ш\.|просп|пр-т|пер|переул|дом|д\.|кв|ком|к.|корп|г\.|город|респ|обл|район|р-он)"

@lru_cache(maxsize=100_000)
def _lemma(token: str) -> str:
    token = token.strip().lower()
    if not _morph or not token:
        return token
    if not any(ch.isalpha() for ch in token):
        return token
    return _morph.parse(token)[0].normal_form

def _build_stop_lemmas() -> set[str]:
    base: set[str] = set()
    if get_stop_words:
        try:
            base |= set(get_stop_words("ru"))
        except Exception:
            pass
    if nltk_stopwords:
        try:
            base |= set(nltk_stopwords.words("russian"))
        except Exception:
            pass
    # доменные леммы
    domain_path = Path(__file__).resolve().parent.parent / "config" / "domain.txt"
    domain_words: set[str] = set()
    try:
        if domain_path.exists():
            for line in domain_path.read_text(encoding="utf-8").splitlines():
                w = line.strip().lower()
                if w:
                    # лемматизируем wpis, чтобы сопоставлялось с леммами текста
                    domain_words.add(w)
    except Exception as e:
        log.warning("Failed to load domain.txt: %s", e)
    
    base |= domain_words
    # в конце возвращаем именно леммы в нижнем регистре
    return { _lemma(w) for w in base if w }

STOP_LEMMAS: set[str] = _build_stop_lemmas()
print("------------------------------------------------", STOP_LEMMAS)

def _token_lemmas(text: str) -> list[str]:
    lems: list[str] = []
    for t in tokenize(text):
        tok = t.text.strip()
        if tok and any(ch.isalpha() for ch in tok):
            lems.append(_lemma(tok))
    return lems

def _is_stop_fragment_by_lemma(fragment: str) -> bool:
    lems = _token_lemmas(fragment)
    if not lems:
        return True
    stop_hits = sum(1 for l in lems if l in STOP_LEMMAS)
    if stop_hits == len(lems):
        return True
    if len(lems) <= 2 and stop_hits >= 1:
        return True
    if stop_hits / max(len(lems), 1) >= 0.8 and len(fragment) < 24:
        return True
    return False


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
    (a0, a1), (b0, b1) = a, b
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0

def _overlaps(a: Tuple[int, int], b: Tuple[int, int], iou_thr: float = 0.5) -> bool:
    (a0, a1), (b0, b1) = a, b
    if (a0 <= b0 and a1 >= b1) or (b0 <= a0 and b1 >= a1):
        return True
    return _iou((a0, a1), (b0, b1)) >= iou_thr


# ---------- Main class ----------
class NatashaNER:
    def __init__(self) -> None:
        self._emb      : Optional[object] = None
        self._ner      : Optional[object] = None
        self._names    : Optional[object] = None
        self._segmenter: Optional[object] = None
        self._addr     : Optional[object] = None

    def _ensure_full_models(self) -> None:
        if any(x is None for x in (Doc, NewsNERTagger, NewsEmbedding, Segmenter)):
            return
        if self._ner is None or self._emb is None or self._segmenter is None:
            try:
                self._emb = self._emb or NewsEmbedding()
                self._ner = self._ner or NewsNERTagger(self._emb)
                self._segmenter = self._segmenter or Segmenter()
                log.debug("Full NER models initialized.")
            except Exception as e:
                log.warning("Failed to init full NER models: %s", e)
                self._emb = self._ner = self._segmenter = None

    def _ensure_names(self) -> None:
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

    def _ensure_addr(self) -> None:
        """Lazily initialize Natasha AddrExtractor."""
        try:
            if self._addr is None:
                morph = MorphVocab()
                self._addr = AddrExtractor(morph)
                log.debug("AddrExtractor initialized.")
        except Exception as e:
            log.warning("Failed to init AddrExtractor: %s", e)
            self._addr = None

    def analyze(self, text: str) -> List[NatashaSpan]:
        spans: List[NatashaSpan] = []

        use_full = os.getenv("REDACTOR_NATASHA_FULL", "0").lower() in {"1","true","yes"}
        if use_full:
            self._ensure_full_models()
        self._ensure_names()

        # --- Full NER path ---
        if use_full and self._ner and Doc and self._segmenter:
            try:
                doc = Doc(text)
                doc.segment(self._segmenter)
                doc.tag_ner(self._ner)
                for span in doc.spans:
                    if span.type in {"PER","ORG"}:
                        # if span.type == "PER":
                        # если все леммы фрагмента стоповые или фрагмент короткий служебный — пропускаем
                        frag = text[span.start:span.stop]
                        # log.info(f"SPAN: {frag} LEMMA: {_lemma(frag)} STOP { _is_stop_fragment_by_lemma(frag) }")
                        if _is_stop_fragment_by_lemma(frag):
                            log.debug(f"Skipping stop fragment: {frag}")
                            continue                        
                        label = "PERSON" if span.type == "PER" else "ORG"
                        score = float(getattr(span, "score", 0.8))
                        spans.append(NatashaSpan(span.start, span.stop, label, score))
                log.debug("Full NER produced %d spans.", len(spans))
            except Exception as e:
                log.warning("Full NER failed: %s", e)

        # --- Fallback: NamesExtractor ---
        if self._names:
            try:
                existing = [(s.start, s.end) for s in spans if s.label == "PERSON"]
                matches = self._names(text)
                added = 0

                for m in matches:
                    s = getattr(m, "start", None)
                    e = getattr(m, "stop",  None)
                    if s is None or e is None:
                        span = getattr(m, "span", None)
                        if span and isinstance(span, (tuple, list)) and len(span)==2:
                            s, e = span
                        else:
                            continue

                    frag = text[s:e]
                    fact = getattr(m, "fact", None)

                    # Лемма-фильтр
                    if _is_stop_fragment_by_lemma(frag):
                        continue

                    # Базовая структурная проверка
                    frag_stripped = frag.strip()
                    if len(frag_stripped) < 4:
                        continue
                    if not frag_stripped[0].isalpha():
                        continue

                    first = getattr(getattr(fact,"first",None),"value",None)
                    last  = getattr(getattr(fact,"last",None),"value",None)
                    middle= getattr(getattr(fact,"middle",None),"value",None)
                    if not (first or last or middle) and len(frag_stripped) < 4:
                        continue

                    if any(_overlaps((s,e), ex) for ex in existing):
                        continue

                    spans.append(NatashaSpan(s, e, "PERSON", 0.75))
                    added += 1

                log.debug("Fallback NamesExtractor added %d PERSON spans.", added)
            except Exception as e:
                log.warning("NamesExtractor failed: %s", e)

        self._ensure_addr()
        if self._addr:
            try:
                matches = self._addr(text)
                added = 0

                for m in matches:
                    s = getattr(m, "start", None)
                    e = getattr(m, "stop",  None)
                    if s is None or e is None:
                        span = getattr(m, "span", None)
                        if span and isinstance(span, (tuple, list)) and len(span)==2:
                            s, e = span
                        else:
                            continue

                    frag = text[s:e]
                    # Лемма-фильтр
                    if _is_stop_fragment_by_lemma(frag) or \
                        not re.search(ADDR_HINTS, frag, flags=re.IGNORECASE):
                        continue

                    spans.append(NatashaSpan(s, e, "ADDRESS", 0.80))
                    added += 1

                log.debug("AddrExtractor added %d ADDRESS spans.", added)
            except Exception as e:
                log.warning("AddrExtractor failed: %s", e)

        return spans

# ---------- End of natasha_ner.py ----------
