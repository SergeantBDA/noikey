from __future__ import annotations

"""Presidio Analyzer initialization and custom recognizers for Russian requisites.

Gracefully degrades when Presidio isn't installed.
"""
from dataclasses import dataclass
from typing import List, Optional, Any, TYPE_CHECKING
import re

try:
    from presidio_analyzer import AnalyzerEngine as _AnalyzerEngine, \
                                  Pattern as _Pattern, \
                                  PatternRecognizer as _PatternRecognizer, \
                                  RecognizerResult as _RecognizerResult, \
                                  RecognizerRegistry as _RecognizerRegistry
                                  # type: ignore[import-not-found]

    from presidio_analyzer.nlp_engine import NlpEngineProvider as _NlpEngineProvider  # type: ignore[import-not-found]
    AnalyzerEngineType: Any = _AnalyzerEngine
    PatternType: Any = _Pattern
    PatternRecognizerBase: Any = _PatternRecognizer
    RecognizerResultType: Any = _RecognizerResult
    NlpEngineProviderType: Any = _NlpEngineProvider
except Exception:  # pragma: no cover - optional dependency
    AnalyzerEngineType = Any  # type: ignore
    PatternType = Any  # type: ignore
    PatternRecognizerBase = object  # type: ignore
    RecognizerResultType = Any  # type: ignore
    NlpEngineProviderType = Any  # type: ignore

from .patterns import (
    validate_inn,
    validate_ogrn,
    validate_kpp,
    validate_bik,
    validate_rs,
    validate_ks,
)


def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s)


class ValidatorPatternRecognizer(PatternRecognizerBase):  # type: ignore[misc]
    """Pattern recognizer with digits-only filtering and custom validation.

    Parameters:
        supported_entity: Label of the entity.
        patterns: List[Pattern] with regexes.
        context: Optional context words to boost score.
        validator: Callable which returns True for valid matched strings.
    """

    def __init__(self, supported_entity: str, patterns, context: Optional[List[str]] = None, validator=None):
        # Ensure these recognizers explicitly support Russian language
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language="ru",
        )
        self._validator = validator

    def validate_result(self, pattern_text: str) -> bool:  # type: ignore[override]
        if self._validator is None:
            return True
        return self._validator(pattern_text)


def build_custom_recognizers():
    if PatternType is Any:
        return []
    recognizers = []

    def pat(regex: str, score: float = 0.6):
        return [PatternType(name=regex, regex=regex, score=score)]

    # INN 10/12
    inn_rec = ValidatorPatternRecognizer(
        supported_entity="INN",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){10}(?:\d[\s-]?){0,2}(?!\d)", 0.7),
        context=["ИНН", "инн"],
        validator=validate_inn,
    )
    recognizers.append(inn_rec)

    # OGRN 13
    ogrn_rec = ValidatorPatternRecognizer(
        supported_entity="OGRN",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){13}(?!\d)", 0.7),
        context=["ОГРН", "огрн"],
        validator=validate_ogrn,
    )
    recognizers.append(ogrn_rec)

    # KPP 9
    kpp_rec = ValidatorPatternRecognizer(
        supported_entity="KPP",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){9}(?!\d)", 0.6),
        context=["КПП", "кпп"],
        validator=validate_kpp,
    )
    recognizers.append(kpp_rec)

    # BIK 9
    bik_rec = ValidatorPatternRecognizer(
        supported_entity="BIK",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){9}(?!\d)", 0.6),
        context=["БИК", "бик"],
        validator=validate_bik,
    )
    recognizers.append(bik_rec)

    # RS20 / KS20
    rs_rec = ValidatorPatternRecognizer(
        supported_entity="RS",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){20}(?!\d)", 0.55),
        context=["р/с", "расчетный счет", "расчётный счёт"],
        validator=validate_rs,
    )
    recognizers.append(rs_rec)

    ks_rec = ValidatorPatternRecognizer(
        supported_entity="KS",
        patterns=pat(r"(?<!\d)(?:\d[\s-]?){20}(?!\d)", 0.55),
        context=["корреспондентский счет", "корр. счет"],
        validator=validate_ks,
    )
    recognizers.append(ks_rec)

    return recognizers


def build_analyzer() -> Optional[Any]:
    """Build Presidio AnalyzerEngine with custom recognizers.

    Returns None if presidio isn't available.
    """
    # If presidio isn't available, return None
    if AnalyzerEngineType is Any:
        return None
    try:
        # Create analyzer without external NLP to keep startup fast; we rely on pattern recognizers
        registry = _RecognizerRegistry(supported_languages=["ru"])
        analyzer = AnalyzerEngineType(registry=registry, supported_languages=["ru"])        
        for rec in build_custom_recognizers():
            analyzer.registry.add_recognizer(rec)
        return analyzer
    except Exception:
        return None


def analyze_text(analyzer: Optional[Any], text: str, min_score: float = 0.5) -> List[dict]:
    """Run analyzer and return span dicts with label, start, end, score, source tag.

    Source tags: "Presidio Custom" for our entities, otherwise "Presidio Default".
    """
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(text=text, language="ru")
    except Exception as e:
        # Fallback: try without language restrictions if registry complains
        try:
            results = analyzer.analyze(text=text, language=None)
        except Exception:
            return []
    spans: List[dict] = []
    for r in results:
        label = r.entity_type  # type: ignore[attr-defined]
        score = float(getattr(r, "score", 0.5))
        if score < min_score:
            continue
        source = "Presidio Default"
        if label in {"INN", "OGRN", "KPP", "BIK", "RS", "KS"}:
            source = "Presidio Custom"
        spans.append({
            "start": int(r.start),  # type: ignore[attr-defined]
            "end": int(r.end),      # type: ignore[attr-defined]
            "label": label,
            "score": score,
            "source": source,
        })
    return spans
