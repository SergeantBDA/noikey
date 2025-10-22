from __future__ import annotations

import re
from redactor.patterns import (
    validate_inn,
    validate_ogrn,
    validate_kpp,
    validate_bik,
    validate_rs,
    validate_ks,
    ROLE_NAMED_PATTERN,
    IN_FACE_PATTERN,
)


def test_inn_validation():
    assert validate_inn("7707083893")  # 10-digit valid
    assert not validate_inn("7707083894")
    assert validate_inn("500100732259")  # 12-digit valid
    assert not validate_inn("500100732250")


def test_ogrn_validation():
    assert validate_ogrn("1027700132195")
    assert not validate_ogrn("1027700132194")


def test_kpp_bik_rs_ks():
    assert validate_kpp("123456789")
    assert validate_bik("044525225")
    assert validate_rs("40702810900000000001")
    assert validate_ks("30101810400000000225")


def test_role_named_pattern():
    text = "Общество ... именуемое в дальнейшем \"Покупатель\" заключило договор"
    m = ROLE_NAMED_PATTERN.search(text)
    assert m is not None


def test_in_face_pattern():
    text = "в лице генерального директора Иванова И.И., действующего на основании Устава"
    m = IN_FACE_PATTERN.search(text)
    assert m is not None
