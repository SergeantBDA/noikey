from __future__ import annotations

"""Presidio Analyzer initialization and custom recognizers for Russian requisites.

Also integrates predefined Email/Phone recognizers and provides optional anonymization.
Gracefully degrades when Presidio isn't installed.
"""

from dataclasses import dataclass
from typing import List, Optional, Any, Dict
import re
import os

try:
    from presidio_analyzer import (
        AnalyzerEngine as _AnalyzerEngine,
        Pattern as _Pattern,
        PatternRecognizer as _PatternRecognizer,
        RecognizerResult as _RecognizerResult,
        RecognizerRegistry as _RecognizerRegistry,
    )  # type: ignore[import-not-found]
    from presidio_analyzer.nlp_engine import NlpEngineProvider as _NlpEngineProvider  # type: ignore[import-not-found]
    from presidio_anonymizer import AnonymizerEngine as _AnonymizerEngine  # type: ignore[import-not-found]
    AnalyzerEngineType: Any = _AnalyzerEngine
    PatternType: Any = _Pattern
    PatternRecognizerBase: Any = _PatternRecognizer
    RecognizerResultType: Any = _RecognizerResult
    RecognizerRegistryType: Any = _RecognizerRegistry
    NlpEngineProviderType: Any = _NlpEngineProvider
    AnonymizerEngineType: Any = _AnonymizerEngine
except Exception:  # pragma: no cover - optional dependency
    AnalyzerEngineType = Any  # type: ignore
    PatternType = Any  # type: ignore
    PatternRecognizerBase = object  # type: ignore
    RecognizerResultType = Any  # type: ignore
    RecognizerRegistryType = Any  # type: ignore
    NlpEngineProviderType = Any  # type: ignore
    AnonymizerEngineType = Any  # type: ignore

from .patterns import (
    validate_inn,
    validate_ogrn,
    validate_kpp,
    validate_bik,
    validate_rs,
    validate_ks,
)


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


def _load_settings() -> Dict[str, Any]:
    """Load optional settings for Presidio integration.

    Order: ENV REDACTOR_SETTINGS -> ./config/settings.yaml -> ./config/settings.json -> defaults.
    """
    import json
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    defaults = {
        "enable_email_recognizer": True,
        "enable_phone_recognizer": True,
        "mask_style": "tag",  # tag | mask | hash
    }
    path = os.getenv("REDACTOR_SETTINGS")
    if not path:
        cwd = os.getcwd()
        ypath = os.path.join(cwd, "config", "settings.yaml")
        jpath = os.path.join(cwd, "config", "settings.json")
        if os.path.isfile(ypath):
            path = ypath
        elif os.path.isfile(jpath):
            path = jpath
    if not path:
        return defaults
    try:
        if path.endswith((".yaml", ".yml")) and yaml is not None:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            return defaults
        cfg = defaults.copy()
        cfg.update({k: data.get(k, v) for k, v in defaults.items()})
        return cfg
    except Exception:
        return defaults


def build_analyzer() -> Optional[Any]:
    """Build Presidio AnalyzerEngine with custom recognizers.

    Returns None if presidio isn't available.
    """
    # If presidio isn't available, return None
    if AnalyzerEngineType is Any:
        return None
    try:
        # Build registry with languages and predefined recognizers (Email/Phone included)
        registry = RecognizerRegistryType(supported_languages=["ru", "en"])  # type: ignore
        registry.load_predefined_recognizers()

        # Try to create an NLP engine; if fails, create analyzer without it
        analyzer: Any
        try:
            provider = NlpEngineProviderType(nlp_configuration={"nlp_engine_name": "spacy", "models": []})
            nlp_engine = provider.create_engine()
            analyzer = AnalyzerEngineType(registry=registry, nlp_engine=nlp_engine, supported_languages=["ru", "en"])  # type: ignore[arg-type]
        except Exception:
            analyzer = AnalyzerEngineType(registry=registry, supported_languages=["ru", "en"])  # type: ignore

        # Load custom recognizers (ru)
        for rec in build_custom_recognizers():
            analyzer.registry.add_recognizer(rec)

        # Apply settings to enable/disable email/phone recognizers
        settings = _load_settings()

        def _remove_by_class_name(name_substr: str) -> None:
            try:
                to_remove = [r for r in analyzer.registry.recognizers if r.__class__.__name__ == name_substr]  # type: ignore[attr-defined]
                for r in to_remove:
                    analyzer.registry.remove_recognizer(r)
            except Exception:
                try:
                    analyzer.registry.remove_recognizer(name_substr)
                except Exception:
                    pass

        if not settings.get("enable_email_recognizer", True):
            _remove_by_class_name("EmailRecognizer")
        if not settings.get("enable_phone_recognizer", True):
            _remove_by_class_name("PhoneRecognizer")
        return analyzer
    except Exception:
        # Last-resort fallback: attempt analyzer with defaults only
        try:
            registry = RecognizerRegistryType(supported_languages=["ru", "en"])  # type: ignore
            registry.load_predefined_recognizers()
            analyzer = AnalyzerEngineType(registry=registry, supported_languages=["ru", "en"])  # type: ignore
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
    except Exception:
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


# -------------- Anonymization (email/phone etc.) --------------

def _build_operators(mask_style: str = "tag") -> Dict[str, Dict[str, Any]]:
    """Build anonymizer operators for email/phone according to mask_style.

    mask_style: 'tag' | 'mask' | 'hash'
    """
    style = (mask_style or "tag").lower()
    if AnonymizerEngineType is Any:
        # If anonymizer isn't available, return replace operators as a fallback
        style = "tag"
    if style == "mask":
        return {
            "EMAIL_ADDRESS": {"type": "mask", "masking_char": "*", "chars_to_mask": "all"},
            "PHONE_NUMBER": {"type": "mask", "masking_char": "*", "chars_to_mask": "all"},
        }
    if style == "hash":
        return {
            "EMAIL_ADDRESS": {"type": "hash", "hash_type": "sha256"},
            "PHONE_NUMBER": {"type": "hash", "hash_type": "sha256"},
        }
    # default tag style
    return {
        "EMAIL_ADDRESS": {"type": "replace", "new_value": "[[ENTITY:EMAIL]]"},
        "PHONE_NUMBER": {"type": "replace", "new_value": "[[ENTITY:PHONE]]"},
    }


# Global anonymizer instance
ANONYMIZER = AnonymizerEngineType() if AnonymizerEngineType is not Any else None  # type: ignore


def analyze_and_anonymize(analyzer: Optional[Any], text: str, min_score: float = 0.5, mask_style: Optional[str] = None) -> str:
    """Analyze with Presidio and anonymize email/phone using configured operators.

    If anonymizer is unavailable, returns original text.
    """
    if analyzer is None or ANONYMIZER is None:
        return text
    settings = _load_settings()
    ops = _build_operators(mask_style or settings.get("mask_style", "tag"))
    try:
        results = analyzer.analyze(text=text, language="ru")
    except Exception:
        results = analyzer.analyze(text=text, language=None)
    filtered = [r for r in results if float(getattr(r, "score", 0.0)) >= float(min_score)]
    anonymized = ANONYMIZER.anonymize(text=text, analyzer_results=filtered, operators=ops)
    return anonymized.text
