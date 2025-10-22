from __future__ import annotations

"""
Regex patterns and rule-based heuristics for Russian contract redaction.

Contains:
- ORG form detection
- Role phrases ("именуемое в дальнейшем ...", "в лице ... действующего на основании ...")
- Subject section detection ("Предмет договора" and keyword-based)
- Validators for INN/OGRN/KPP/BIK/RS/KS
"""
from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, List, Optional, Sequence, Tuple

# --- Entity label constants ---
LABEL_ORG = "ORG"
LABEL_ROLE = "ROLE"
LABEL_SUBJECT = "SUBJECT"

# --- Compiled regexes ---
ORG_PATTERN = re.compile(
    r"\b(?:ООО|АО|ПАО|ЗАО|ОАО|ГУП|МУП|ИП|ФГБУ|АНО|НКО|ТСЖ)\s+[«\"']?[А-ЯЁA-Z0-9][^\n\"»]{1,100}[»\"']?",
    re.U,
)

ROLE_NAMED_PATTERN = re.compile(
    r"именуем\w*\s+в\s+дальнейшем\s+[«\"']?([А-ЯЁA-Z][^»\"\n]{1,40})[»\"']?",
    re.I | re.U,
)

IN_FACE_PATTERN = re.compile(
    r"в\s+лице\s+(?P<role>[А-ЯЁA-Z][а-яё]+(?:\s+[А-ЯЁA-Z]\.){1,2}|[А-ЯЁA-Z][а-яё]+\s+[А-ЯЁA-Z][а-яё]+\s+[А-ЯЁA-Z][а-яё]+).*?действующ\w*\s+на\s+основан\w+",
    re.I | re.U | re.S,
)

SUBJECT_HEADER_PATTERN = re.compile(
    r"^\s*(ПРЕДМЕТ\s+ДОГОВОРА|Предмет\s+договора)\s*$",
    re.M | re.U,
)

SUBJECT_KEYWORDS = (
    "обязуется",
    "предметом является",
    "поставить",
    "оказать",
    "выполнить",
    "предоставить",
)

UPPERCASE_SECTION_RE = re.compile(r"^\s*[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9\s\.-]{2,}$", re.M)
NUMBERED_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s*\.?.{0,60}$", re.M)


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


# --- Validators for requisites ---

def _digits_only(s: str) -> str:
    return re.sub(r"\D+", "", s)


def validate_inn(inn: str) -> bool:
    """Validate Russian INN (10 or 12 digits) with checksum.

    Algorithm source: FNS checksum rules. For 10-digit (org) use weights [2,4,10,3,5,9,4,6,8],
    for 12-digit (individual) calculate 11th and 12th digits with respective weights.
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
    """Validate OGRN (13 digits).

    Rule: last digit equals (first 12 digits as int) % 11 % 10.
    """
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


# --- Regex-driven detectors ---

def find_orgs(text: str) -> List[RegexSpan]:
    spans: List[RegexSpan] = []
    for m in ORG_PATTERN.finditer(text):
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ORG, 0.85))
    return spans


def find_roles(text: str) -> List[RegexSpan]:
    spans: List[RegexSpan] = []
    # "именуемое в дальнейшем ..."
    for m in ROLE_NAMED_PATTERN.finditer(text):
        spans.append(RegexSpan(m.start(), m.end(), LABEL_ROLE, 0.9))
    # "в лице ... действующего на основании ..." (treat whole phrase as ROLE span)
    for m in IN_FACE_PATTERN.finditer(text):
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
    spans: List[RegexSpan] = []
    # Header-based
    for m in SUBJECT_HEADER_PATTERN.finditer(text):
        start = m.start()
        end = _next_section_start(text, m.end())
        # Trim to max_size by paragraph boundaries
        chunk = text[start:end]
        if len(chunk) > max_size:
            cut = chunk[:max_size]
            # try to cut at paragraph boundary
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
    for kw in SUBJECT_KEYWORDS:
        i = lower.find(kw)
        if i != -1:
            # expand to surrounding paragraph
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
