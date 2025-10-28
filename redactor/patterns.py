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
LABEL_ORG     = "ORG"
LABEL_ROLE    = "ROLE"
LABEL_SUBJECT = "SUBJECT"
LABEL_EMAIL   = "EMAIL"
LABEL_URL     = "URL"
LABEL_INN     = "INN"
LABEL_OGRN    = "OGRN"
LABEL_KPP     = "KPP"
LABEL_BIK     = "BIK"
LABEL_RS      = "RS"
LABEL_KS      = "KS"
LABEL_ADDR    = "ADDR"

log = logging.getLogger(__name__)

class FlagsModel(BaseModel):
    org_pattern: List[str] = []
    role_named_pattern: List[str] = []
    in_face_pattern: List[str] = []
    subject_header_pattern: List[str] = []
    email_pattern: List[str] = []
    url_pattern: List[str] = []
    inn_pattern: List[str] = []
    ogrn_pattern: List[str] = []
    kpp_pattern: List[str] = []
    bik_pattern: List[str] = []
    rs_pattern: List[str] = []
    ks_pattern: List[str] = []
    addr_window_pattern: List[str] = []
    addr_street_pattern: List[str] = []
    addr_house_pattern: List[str] = []

class RegexModel(BaseModel):
    org_pattern:  Optional[str] = None
    role_named_pattern: Optional[str] = None
    in_face_pattern: Optional[str] = None
    subject_header_pattern: Optional[str] = None
    email_pattern: Optional[str] = None
    url_pattern: Optional[str] = None
    inn_pattern: Optional[str] = None
    ogrn_pattern: Optional[str] = None
    kpp_pattern: Optional[str] = None
    bik_pattern: Optional[str] = None
    rs_pattern: Optional[str] = None
    ks_pattern: Optional[str] = None
    addr_window_pattern: Optional[str] = None
    addr_street_pattern: Optional[str] = None
    addr_house_pattern: Optional[str] = None

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
    org_pattern:            Optional[re.Pattern] = None
    role_named_pattern:     Optional[re.Pattern] = None
    in_face_pattern:        Optional[re.Pattern] = None
    subject_header_pattern: Optional[re.Pattern] = None
    subject_keywords:       Tuple[str, ...]      = ()
    email_pattern:          Optional[re.Pattern] = None
    url_pattern:            Optional[re.Pattern] = None
    inn_pattern:            Optional[re.Pattern] = None
    ogrn_pattern:           Optional[re.Pattern] = None
    kpp_pattern:            Optional[re.Pattern] = None
    bik_pattern:            Optional[re.Pattern] = None
    rs_pattern:             Optional[re.Pattern] = None
    ks_pattern:             Optional[re.Pattern] = None
    addr_window_pattern: Optional[re.Pattern] = None
    addr_street_pattern: Optional[re.Pattern] = None
    addr_house_pattern: Optional[re.Pattern] = None

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
        org   = re.compile(cfg.regex.org_pattern,            _flags_to_re(cfg.flags.org_pattern))
        role  = re.compile(cfg.regex.role_named_pattern,     _flags_to_re(cfg.flags.role_named_pattern))
        inf   = re.compile(cfg.regex.in_face_pattern,        _flags_to_re(cfg.flags.in_face_pattern))
        subj  = re.compile(cfg.regex.subject_header_pattern, _flags_to_re(cfg.flags.subject_header_pattern))
        email = re.compile(cfg.regex.email_pattern,          _flags_to_re(cfg.flags.email_pattern))
        url   = re.compile(cfg.regex.url_pattern,            _flags_to_re(cfg.flags.url_pattern))
        inn   = re.compile(cfg.regex.inn_pattern,            _flags_to_re(cfg.flags.inn_pattern))
        ogrn  = re.compile(cfg.regex.ogrn_pattern,           _flags_to_re(cfg.flags.ogrn_pattern))
        kpp   = re.compile(cfg.regex.kpp_pattern,            _flags_to_re(cfg.flags.kpp_pattern))
        bik   = re.compile(cfg.regex.bik_pattern,            _flags_to_re(cfg.flags.bik_pattern))
        rs    = re.compile(cfg.regex.rs_pattern,             _flags_to_re(cfg.flags.rs_pattern))
        ks    = re.compile(cfg.regex.ks_pattern,             _flags_to_re(cfg.flags.ks_pattern))
        addr_window = re.compile(cfg.regex.addr_window_pattern, _flags_to_re(cfg.flags.addr_window_pattern))
        addr_street = re.compile(cfg.regex.addr_street_pattern, _flags_to_re(cfg.flags.addr_street_pattern))
        addr_house = re.compile(cfg.regex.addr_house_pattern, _flags_to_re(cfg.flags.addr_house_pattern))

    except re.error as e:
        # Identify which pattern failed by naive check
        # Re-compile each to isolate
        for name, pat, fl in [
            ("org_pattern", cfg.regex.org_pattern, cfg.flags.org_pattern),
            ("role_named_pattern", cfg.regex.role_named_pattern, cfg.flags.role_named_pattern),
            ("in_face_pattern", cfg.regex.in_face_pattern, cfg.flags.in_face_pattern),
            ("subject_header_pattern", cfg.regex.subject_header_pattern, cfg.flags.subject_header_pattern),
            ("email_pattern", cfg.regex.email_pattern, cfg.flags.email_pattern),
            ("url_pattern", cfg.regex.url_pattern, cfg.flags.url_pattern),
            ("inn_pattern", cfg.regex.inn_pattern, cfg.flags.inn_pattern),
            ("ogrn_pattern", cfg.regex.ogrn_pattern, cfg.flags.ogrn_pattern),
            ("kpp_pattern", cfg.regex.kpp_pattern, cfg.flags.kpp_pattern),
            ("bik_pattern", cfg.regex.bik_pattern, cfg.flags.bik_pattern),
            ("rs_pattern", cfg.regex.rs_pattern, cfg.flags.rs_pattern),
            ("ks_pattern", cfg.regex.ks_pattern, cfg.flags.ks_pattern),
            ("addr_window_pattern", cfg.regex.addr_window_pattern, cfg.flags.addr_window_pattern),
            ("addr_street_pattern", cfg.regex.addr_street_pattern, cfg.flags.addr_street_pattern),
            ("addr_house_pattern", cfg.regex.addr_house_pattern, cfg.flags.addr_house_pattern),            
        ]:
            try:
                re.compile(pat, _flags_to_re(fl))
            except re.error as e2:
                log.error("Failed to compile regex '%s': %s", name, e2)
                raise
        # If not isolated, re-raise original
        raise
    subject_keywords = tuple(cfg.keywords.get("subject_keywords", []))
    return PatternBundle(org, role, inf, subj, subject_keywords,
                         email, url, inn, ogrn, kpp, bik, rs, ks, addr_window, addr_street, addr_house)


def load_patterns(config_path: Optional[str] = None) -> PatternBundle:
    path, src = _resolve_config_path(config_path)
    if path is None:
        log.info("Patterns config not found, using defaults (source=%s)", src)
        return None
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
        if _CACHE is None:
            raise RuntimeError("Patterns bundle is not initialized")
        return _CACHE

def reload_patterns(config_path: Optional[str] = None) -> PatternBundle:
    global _CACHE, _CACHE_PATH
    with _LOCK:
        _CACHE = None
        _CACHE_PATH = config_path
        log.info("Hot-reload patterns (path=%s)", config_path or "auto")
        bundle = load_patterns(config_path)
        if bundle is None:
            raise RuntimeError("Failed to load patterns on reload")
        _CACHE = bundle
        return bundle
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

    # 1) Фразы вида "в лице ... действующ... на основании ..."
    for m in pats.in_face_pattern.finditer(text):
        # Контекстная метка роли (как было раньше)
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ROLE, 0.92))

    # 2) Доверенности: маскируем ТОЛЬКО номер и дату
    for m in pats.role_named_pattern.finditer(text):
        # при желании можно оставить общий контекст роли, но с низким скором:
        #spans.append(RegexSpan(m.start(), m.end(), LABEL_ROLE, 0.30, extra_label="POA_CONTEXT"))

        # номер доверенности
        if "num" in m.groupdict() and m.group("num"):
            try:
                s_num = m.start("num")
                e_num = m.end("num")
                spans.append(RegexSpan(s_num, e_num, LABEL_ROLE, 0.97, extra_label="POA_NUM"))
            except IndexError:
                pass

        # дата доверенности
        if "date" in m.groupdict() and m.group("date"):
            try:
                s_dt = m.start("date")
                e_dt = m.end("date")
                spans.append(RegexSpan(s_dt, e_dt, LABEL_ROLE, 0.97, extra_label="POA_DATE"))
            except IndexError:
                pass

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

def find_emails(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.email_pattern:
        for m in pats.email_pattern.finditer(text):
            spans.append(RegexSpan(m.start(), m.end(), LABEL_EMAIL, 0.95))
    return spans


def find_urls(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.url_pattern:
        for m in pats.url_pattern.finditer(text):
            spans.append(RegexSpan(m.start(), m.end(), LABEL_URL, 0.95))
    return spans

def find_inn(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.inn_pattern:
        for m in pats.inn_pattern.finditer(text):
            raw = m.group(0)
            if validate_inn(raw):  # ← строгая проверка контрольной суммы/длины
                spans.append(RegexSpan(m.start(), m.end(), LABEL_INN, 0.98))
    return spans

def find_ogrn(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.ogrn_pattern:
        for m in pats.ogrn_pattern.finditer(text):
            raw = m.group(0)
            if validate_ogrn(raw):
                spans.append(RegexSpan(m.start(), m.end(), LABEL_OGRN, 0.98))
    return spans

def find_kpp(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.kpp_pattern:
        for m in pats.kpp_pattern.finditer(text):
            raw = m.group(0)
            if validate_kpp(raw):
                spans.append(RegexSpan(m.start(), m.end(), LABEL_KPP, 0.95))
    return spans

def find_bik(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.bik_pattern:
        for m in pats.bik_pattern.finditer(text):
            raw = m.group(0)
            if validate_bik(raw):
                spans.append(RegexSpan(m.start(), m.end(), LABEL_BIK, 0.95))
    return spans

def find_rs(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.rs_pattern:
        for m in pats.rs_pattern.finditer(text):
            raw = m.group(0)
            if validate_rs(raw):
                spans.append(RegexSpan(m.start(), m.end(), LABEL_RS, 0.96))
    return spans

def find_ks(text: str) -> List[RegexSpan]:
    pats = get_patterns()
    spans: List[RegexSpan] = []
    if pats.ks_pattern:
        for m in pats.ks_pattern.finditer(text):
            raw = m.group(0)
            if validate_ks(raw):
                spans.append(RegexSpan(m.start(), m.end(), LABEL_KS, 0.96))
    return spans

def find_addresses_strict(text: str) -> List[RegexSpan]:
    """
    Берём только те «окна», где в пределах 40–120 символов есть >=2 адресных маркера,
    и внутри встречается хотя бы улица/улицетип ИЛИ дом/корп./стр. с номером.
    Возвращаем цельный фрагмент окна (слегка расширенный вправо до ближайшего ';' или '.').
    """
    spans: List[RegexSpan] = []
    pats = get_patterns()  # существует в модуле
    win_rx   = pats.addr_window_pattern
    street_rx= pats.addr_street_pattern
    house_rx = pats.addr_house_pattern
    if not (win_rx and street_rx and house_rx):
        return spans  # конфиг не подключён — тихо выходим

    tnorm = text

    for m in win_rx.finditer(tnorm):
        frag = tnorm[m.start():m.end()]
        if not (street_rx.search(frag) or house_rx.search(frag)):
            continue
        # чуть расширим границы справа, чтобы не обрубать номер
        left  = max(0, m.start() - 40)
        right = min(len(tnorm), m.end() + 40)
        cut = re.search(r"[.;](?:\s|$)", tnorm[m.end():right])
        if cut:
            right = m.end() + cut.start() + 1
        raw_slice = tnorm[left:right]
        # приблизительное сопоставление с оригиналом
        raw_pos = text.find(raw_slice)
        if raw_pos == -1:
            raw_pos = left
        spans.append(RegexSpan(raw_pos, raw_pos + len(raw_slice), LABEL_ADDR, 0.93))
    return spans

def regex_find_spans(text: str) -> List[RegexSpan]:
    spans: List[RegexSpan] = []
    spans.extend(find_orgs(text))
    spans.extend(find_roles(text))
    spans.extend(find_subjects(text))
    spans.extend(find_emails(text))
    spans.extend(find_urls(text))
    spans.extend(find_inn(text))
    spans.extend(find_ogrn(text))
    spans.extend(find_kpp(text))
    spans.extend(find_bik(text))
    spans.extend(find_rs(text))
    spans.extend(find_ks(text)) 
    spans.extend(find_addresses_strict(text))  
    return spans