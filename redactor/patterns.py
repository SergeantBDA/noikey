from __future__ import annotations

"""
Pattern management and regex/rule heuristics for Russian contract redaction.

Usage example:
    from redactor.patterns import get_patterns
    pats = get_patterns()
    if pats.org_pattern.search(text):
        ...

This module provides:
- Loading patterns/keywords from YAML/JSON config with validation (pydantic)
- Compilation with flags, caching, thread safety, hot-reload
- Backward-compatible globals (deprecated) for existing imports
- Regex-based detectors for ORG/ROLE/SUBJECT using the loaded bundle
- Validators for requisites (INN/OGRN/etc.) remain here

Performance: get_patterns() caches a compiled PatternBundle. Avoid calling it in tight loops; store the bundle reference and reuse.
"""

import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Mapping
import warnings

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

from pydantic import BaseModel, Field, validator

# --- Entity label constants ---
LABEL_ORG = "ORG"
LABEL_ROLE = "ROLE"
LABEL_SUBJECT = "SUBJECT"

log = logging.getLogger(__name__)


# Default config (fallback)
DEFAULT_CONFIG = {
    "version": 1,
    "regex": {
        "org_pattern": r"(?:ООО|АО|ПАО|ЗАО|ОАО|ГУП|МУП|ИП|ФГБУ|АНО|НКО|ТСЖ)\s+[«\"']?[А-ЯЁA-Z0-9][^\"»\n]{1,100}[»\"']?",
        "role_named_pattern": r"именуем\w*\s+в\s+дальнейшем\s+[«\"']?\s*([А-ЯЁA-Z][^»\"\n]{1,40})[»\"']?",
        "in_face_pattern": r"в\s+лице\s+[^,\n]+?,\s*действующ\w*\s+на\s+основан\w+",
        "subject_header_pattern": r"(?im)^\s*(ПРЕДМЕТ\s+ДОГОВОРА|Предмет\s+договора)\s*$",
    },
    "flags": {
        "org_pattern": ["U"],
        "role_named_pattern": ["I", "U"],
        "in_face_pattern": ["I", "U"],
        "subject_header_pattern": ["I", "M", "U"],
    },
    "keywords": {
        "subject_keywords": [
            "обязуется",
            "предметом является",
            "оказать",
            "выполнить",
            "поставить",
        ]
    },
}


class FlagsModel(BaseModel):
    org_pattern: List[str] = []
    role_named_pattern: List[str] = []
    in_face_pattern: List[str] = []
    subject_header_pattern: List[str] = []


class RegexModel(BaseModel):
    org_pattern: str
    role_named_pattern: str
    in_face_pattern: str
    subject_header_pattern: str


class PatternsConfig(BaseModel):
    version: int = 1
    regex: RegexModel
    flags: FlagsModel = FlagsModel()
    keywords: Mapping[str, List[str]] = Field(default_factory=dict)

    class Config:
        extra = "ignore"  # ignore unknowns, we'll warn manually

    @validator("version")
    def _check_version(cls, v: int) -> int:  # noqa: N805
        if v != 1:
            log.warning("Unsupported patterns config version %s; attempting to proceed.", v)
        return v


# --- Pattern bundle ---
@dataclass(frozen=True)
class PatternBundle:
    org_pattern: re.Pattern
    role_named_pattern: re.Pattern
    in_face_pattern: re.Pattern
    subject_header_pattern: re.Pattern
    subject_keywords: Tuple[str, ...]


_CACHE: Optional[PatternBundle] = None
_CACHE_PATH: Optional[str] = None
_LOCK = threading.Lock()


def _flags_to_re(flags: List[str]) -> int:
    mapping = {
        "I": re.IGNORECASE,
        "M": re.MULTILINE,
        "S": re.DOTALL,
        "U": re.UNICODE,
        "X": re.VERBOSE,
    }
    f = 0
    unknown: List[str] = []
    for k in flags or []:
        if k in mapping:
            f |= mapping[k]
        else:
            unknown.append(k)
    if unknown:
        log.warning("Unknown regex flags in patterns config: %s", ",".join(unknown))
    # Ensure UNICODE by default
    if not any(k == "U" for k in flags or []):
        f |= re.UNICODE
    return f


# --- Validators for requisites (kept here for backward-compat) ---

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s)


def validate_inn(inn: str) -> bool:
    """Validate Russian INN (10 or 12 digits) with checksum.

    For 10-digit: weights [2,4,10,3,5,9,4,6,8], control = sum % 11 % 10 equals digit 10.
    For 12-digit: compute digits 11 and 12 with respective weights.
    """
    digits = _digits_only(inn)
    if len(digits) == 10:
        coeffs = [2, 4, 10, 3, 5, 9, 4, 6, 8]
        c = sum(int(d) * w for d, w in zip(digits[:9], coeffs)) % 11 % 10
        return c == int(digits[9])
    if len(digits) == 12:
        coeffs11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        coeffs12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8, 0]
        n11 = sum(int(d) * w for d, w in zip(digits[:10], coeffs11)) % 11 % 10
        n12 = sum(int(d) * w for d, w in zip(digits[:11], coeffs12)) % 11 % 10
        return n11 == int(digits[10]) and n12 == int(digits[11])
    return False


def validate_ogrn(ogrn: str) -> bool:
    """Validate OGRN (13 digits). Control digit = (int(first 12) % 11) % 10."""
    digits = _digits_only(ogrn)
    if len(digits) != 13:
        return False
    base = int(digits[:12])
    ctrl = (base % 11) % 10
    return ctrl == int(digits[12])


def validate_kpp(kpp: str) -> bool:
    digits = _digits_only(kpp)
    return len(digits) == 9


def validate_bik(bik: str) -> bool:
    digits = _digits_only(bik)
    return len(digits) == 9


def validate_rs(account: str) -> bool:
    digits = _digits_only(account)
    return len(digits) == 20


def validate_ks(account: str) -> bool:
    digits = _digits_only(account)
    return len(digits) == 20


def _load_file(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    if ext in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed but a YAML patterns file was provided")
        return yaml.safe_load(data) or {}
    if ext == ".json":
        return json.loads(data)
    raise ValueError(f"Unsupported patterns config format: {ext}")


def _resolve_config_path(explicit: Optional[str]) -> Tuple[Optional[str], str]:
    # Priority: explicit -> ENV -> ./config/patterns.yaml -> ./config/patterns.json -> fallback
    if explicit:
        return explicit, "EXPLICIT"
    env_var = os.getenv("REDactor_PATTERNS") or os.getenv("REDACTOR_PATTERNS")
    if env_var:
        return env_var, "ENV"
    yaml_path = os.path.join(os.getcwd(), "config", "patterns.yaml")
    json_path = os.path.join(os.getcwd(), "config", "patterns.json")
    if os.path.isfile(yaml_path):
        return yaml_path, "DEFAULT_PATH"
    if os.path.isfile(json_path):
        return json_path, "DEFAULT_PATH"
    return None, "FALLBACK"


def _compile(cfg: PatternsConfig) -> PatternBundle:
    try:
        org = re.compile(cfg.regex.org_pattern, _flags_to_re(cfg.flags.org_pattern))
        print(cfg.regex.org_pattern)
        role = re.compile(cfg.regex.role_named_pattern, _flags_to_re(cfg.flags.role_named_pattern))
        inf = re.compile(cfg.regex.in_face_pattern, _flags_to_re(cfg.flags.in_face_pattern))
        subj = re.compile(cfg.regex.subject_header_pattern, _flags_to_re(cfg.flags.subject_header_pattern))
    except re.error as e:
        # Identify which pattern failed by naive check
        # Re-compile each to isolate
        for name, pat, fl in [
            ("org_pattern", cfg.regex.org_pattern, cfg.flags.org_pattern),
            ("role_named_pattern", cfg.regex.role_named_pattern, cfg.flags.role_named_pattern),
            ("in_face_pattern", cfg.regex.in_face_pattern, cfg.flags.in_face_pattern),
            ("subject_header_pattern", cfg.regex.subject_header_pattern, cfg.flags.subject_header_pattern),
        ]:
            try:
                re.compile(pat, _flags_to_re(fl))
            except re.error as e2:
                log.error("Failed to compile regex '%s': %s", name, e2)
                raise
        # If not isolated, re-raise original
        raise
    subject_keywords = tuple(cfg.keywords.get("subject_keywords", []))
    return PatternBundle(org, role, inf, subj, subject_keywords)


def load_patterns(config_path: Optional[str] = None) -> PatternBundle:
    path, src = _resolve_config_path(config_path)
    if path is None:
        log.info("Patterns config not found, using defaults (source=%s)", src)
        data = DEFAULT_CONFIG
    else:
        log.info("Loading patterns config from %s (source=%s)", path, src)
        try:
            data = _load_file(path)
        except Exception as e:
            log.error("Failed to load patterns config '%s': %s", path, e)
            raise
    # Warn on unknown top-level keys
    allowed_top = {"version", "regex", "flags", "keywords"}
    unknown_top = set(data.keys()) - allowed_top
    if unknown_top:
        log.warning("Unknown top-level keys in patterns config: %s", ", ".join(sorted(unknown_top)))

    cfg = PatternsConfig(**data)
    log.info("Patterns schema version: %s", cfg.version)
    bundle = _compile(cfg)
    return bundle


def get_patterns() -> PatternBundle:
    global _CACHE
    with _LOCK:
        if _CACHE is None:
            _CACHE = load_patterns(_CACHE_PATH)
        return _CACHE


def reload_patterns(config_path: Optional[str] = None) -> PatternBundle:
    global _CACHE, _CACHE_PATH
    with _LOCK:
        _CACHE = None
        _CACHE_PATH = config_path
        log.info("Hot-reload patterns (path=%s)", config_path or "auto")
        bundle = load_patterns(config_path)
        _CACHE = bundle
        # Update deprecated globals on reload for better backward-compat
        _update_deprecated_globals(bundle)
        return bundle


# --- Deprecated globals (backward-compat) ---

def _update_deprecated_globals(bundle: "PatternBundle") -> None:
    globals()["ORG_PATTERN"] = bundle.org_pattern
    globals()["ROLE_NAMED_PATTERN"] = bundle.role_named_pattern
    globals()["IN_FACE_PATTERN"] = bundle.in_face_pattern
    globals()["SUBJECT_HEADER_PATTERN"] = bundle.subject_header_pattern
    globals()["SUBJECT_KEYWORDS"] = list(bundle.subject_keywords)


# Initialize deprecated proxies
_BUNDLE = get_patterns()
_update_deprecated_globals(_BUNDLE)
warnings.warn(
    "Importing ORG_PATTERN/ROLE_NAMED_PATTERN/IN_FACE_PATTERN/SUBJECT_HEADER_PATTERN/SUBJECT_KEYWORDS from redactor.patterns is deprecated. "
    "They are now loaded from configuration. Prefer get_patterns().",
    DeprecationWarning,
    stacklevel=2,
)


# --- Regex-driven detectors using dynamic patterns ---

@dataclass(frozen=True)
class RegexSpan:
    """Simple span result from regex/rules.

    Attributes:
        start: Start index (inclusive) in the input text.
        end: End index (exclusive).
        label: Entity type label.
        score: Confidence score in [0,1].
        source: Source tag (e.g., "Regex").
        extra_label: Optional secondary label when masking a combined span (e.g., ROLE + PERSON).
    """

    start: int
    end: int
    label: str
    score: float
    source: str = "Regex"
    extra_label: Optional[str] = None


# Section heading helpers remain static (not configurable for now)
UPPERCASE_SECTION_RE = re.compile(r"^\s*[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\s\.-]{2,}$", re.M)
NUMBERED_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s*\.?.{0,60}$", re.M)


def find_orgs(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    for m in pats.org_pattern.finditer(text):
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ORG, 0.85))
    return spans


def find_roles(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    for m in pats.role_named_pattern.finditer(text):
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ROLE, 0.9))
    for m in pats.in_face_pattern.finditer(text):
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ROLE, 0.92))
    return spans


def _next_section_start(text: str, from_pos: int) -> int:
    next_upper = UPPERCASE_SECTION_RE.search(text, pos=from_pos)
    next_num = NUMBERED_SECTION_RE.search(text, pos=from_pos)
    candidates = [m.start() for m in (next_upper, next_num) if m]
    return min(candidates) if candidates else len(text)


def find_subjects(text: str, max_size: int = 2000) -> List[RegexSpan]:
    """Find subject sections by header or by keyword heuristics.

    If header found, capture until next section-like header. Otherwise,
    capture paragraph(s) around keyword indicators, limited by max_size.
    """
    pats = get_patterns()
    spans: List[RegexSpan] = []
    # Header-based
    for m in pats.subject_header_pattern.finditer(text):
        start = m.start()
        end = _next_section_start(text, m.end())
        # Trim to max_size by paragraph boundaries
        chunk = text[start:end]
        if len(chunk) > max_size:
            cut = chunk[:max_size]
            p = cut.rfind("\n\n")
            if p > 0:
                end = start + p
            else:
                end = start + max_size
        spans.append(RegexSpan(start, end, LABEL_SUBJECT, 0.95))

    if spans:
        return spans

    # Keyword-based
    lower = text.lower()
    for kw in pats.subject_keywords:
        i = lower.find(kw)
        if i != -1:
            start = lower.rfind("\n\n", 0, i)
            start = 0 if start == -1 else start + 2
            end = lower.find("\n\n", i)
            end = len(text) if end == -1 else end
            if end - start > max_size:
                end = start + max_size
                p = text.rfind("\n\n", start, end)
                if p > start:
                    end = p
            spans.append(RegexSpan(start, end, LABEL_SUBJECT, 0.7))
    return spans


def regex_find_spans(text: str) -> List[RegexSpan]:
    """Run all regex-based detectors and return spans."""
    spans: List[RegexSpan] = []
    spans.extend(find_orgs(text))
    spans.extend(find_roles(text))
    spans.extend(find_subjects(text))
    return spans
