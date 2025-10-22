from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from redactor.patterns import (
    load_patterns,
    get_patterns,
    reload_patterns,
    _flags_to_re,  # type: ignore
    PatternBundle,
)


def test_default_load_when_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Ensure no config present
    monkeypatch.delenv("REDactor_PATTERNS", raising=False)
    monkeypatch.delenv("REDACTOR_PATTERNS", raising=False)
    monkeypatch.chdir(tmp_path)
    bundle = load_patterns()
    assert isinstance(bundle, PatternBundle)
    assert bundle.subject_keywords  # has defaults


def test_yaml_and_json_equivalent(tmp_path: Path):
    # Prepare YAML
    (tmp_path / "config").mkdir()
    yaml_path = tmp_path / "config" / "patterns.yaml"
    yaml_path.write_text(
        """
version: 1
regex:
  org_pattern: '(?:ООО|АО)\s+\"X\"'
  role_named_pattern: 'именуем\\w*\s+в\s+дальнейшем\s+\"Роль\"'
  in_face_pattern: 'в\s+лице\s+[^,\n]+?,\s*действующ\\w*\s+на\s+основан\\w+'
  subject_header_pattern: '(?im)^\s*(ПРЕДМЕТ\s+ДОГОВОРА)\s*$'
flags:
  org_pattern: ['U']
  role_named_pattern: ['I','U']
  in_face_pattern: ['I','U']
  subject_header_pattern: ['I','M','U']
keywords:
  subject_keywords:
    - 'обязуется'
        """,
        encoding="utf-8",
    )
    # Load YAML
    os.chdir(tmp_path)
    b_yaml = load_patterns()

    # Prepare JSON equivalent
    (tmp_path / "config").mkdir(exist_ok=True)
    json_path = tmp_path / "config" / "patterns.json"
    json_path.write_text(
        json.dumps(
            {
                "version": 1,
                "regex": {
                    "org_pattern": r"(?:ООО|АО)\s+\"X\"",
                    "role_named_pattern": r"именуем\w*\s+в\s+дальнейшем\s+\"Роль\"",
                    "in_face_pattern": r"в\s+лице\s+[^,\n]+?,\s*действующ\w*\s+на\s+основан\w+",
                    "subject_header_pattern": r"(?im)^\s*(ПРЕДМЕТ\s+ДОГОВОРА)\s*$",
                },
                "flags": {
                    "org_pattern": ["U"],
                    "role_named_pattern": ["I", "U"],
                    "in_face_pattern": ["I", "U"],
                    "subject_header_pattern": ["I", "M", "U"],
                },
                "keywords": {"subject_keywords": ["обязуется"]},
            }
        ),
        encoding="utf-8",
    )
    b_json = load_patterns(str(json_path))

    # Sanity: both compiled patterns exist
    assert b_yaml.org_pattern.pattern == b_json.org_pattern.pattern
    assert b_yaml.subject_keywords == b_json.subject_keywords


def test_bad_regex_raises(tmp_path: Path):
    (tmp_path / "config").mkdir()
    bad_path = tmp_path / "config" / "patterns.yaml"
    bad_path.write_text(
        """
version: 1
regex:
  org_pattern: '([unclosed'
  role_named_pattern: 'x'
  in_face_pattern: 'x'
  subject_header_pattern: 'x'
flags: {}
keywords: {}
        """,
        encoding="utf-8",
    )
    os.chdir(tmp_path)
    with pytest.raises(Exception):
        load_patterns()


def test_flags_converter_implies_unicode():
    f = _flags_to_re(["I"])  # no U
    # Should include IGNORECASE and UNICODE
    import re

    assert f & re.IGNORECASE
    assert f & re.UNICODE


def test_hot_reload_changes_patterns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    p1 = tmp_path / "config" / "patterns.json"
    p1.write_text(
        json.dumps(
            {
                "version": 1,
                "regex": {
                    "org_pattern": r"ООО\s+\"X\"",
                    "role_named_pattern": r"именуем\w*",
                    "in_face_pattern": r"в\s+лице",
                    "subject_header_pattern": r"(?im)^\s*ПРЕДМЕТ\s+ДОГОВОРА\s*$",
                },
                "flags": {},
                "keywords": {"subject_keywords": ["обязуется"]},
            }
        ),
        encoding="utf-8",
    )
    b1 = reload_patterns(str(p1))
    assert "ООО" in b1.org_pattern.pattern

    # Change and reload
    p1.write_text(
        json.dumps(
            {
                "version": 1,
                "regex": {
                    "org_pattern": r"АО\s+\"Y\"",
                    "role_named_pattern": r"именуем\w*",
                    "in_face_pattern": r"в\s+лице",
                    "subject_header_pattern": r"(?im)^\s*ПРЕДМЕТ\s+ДОГОВОРА\s*$",
                },
                "flags": {},
                "keywords": {"subject_keywords": ["обязуется"]},
            }
        ),
        encoding="utf-8",
    )
    b2 = reload_patterns(str(p1))
    assert "АО" in b2.org_pattern.pattern
