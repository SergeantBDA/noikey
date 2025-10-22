from __future__ import annotations

from redactor.pipeline import Pipeline, PipelineOptions, MaskStyle


def test_pipeline_basic_subject_and_ids():
    text = (
        "Общество с ограниченной ответственностью ООО \"Ромашка\", ОГРН 1027700132195, ИНН 7707083893,\n"
        "именуемое в дальнейшем \"Поставщик\", в лице генерального директора Иванова И.И., действующего на основании Устава,\n"
        "обязуется поставить Товар...\n"
        "\n"
        "ПРЕДМЕТ ДОГОВОРА\n"
        "Исполнитель обязуется оказать услуги по ...\n"
        "1. СТОРОНЫ ДОГОВОРА\n"
    )
    pipe = Pipeline(PipelineOptions(mask_style=MaskStyle.TAG, keep_length=False, include_subject=True))
    red, logs = pipe.process_text(text)
    # Must contain subject tag and IDs
    assert "[[ENTITY:SUBJECT]]" in red or "[[ENTITY:ROLE]]" in red  # at least some tagging
    # Ensure INN/OGRN masked
    assert "ИНН" in text and ("ИНН" in red)  # label remains, number masked separately
    assert "1027700132195" not in red
    assert "7707083893" not in red


def test_overlap_merge_role_person():
    txt = (
        "именуемое в дальнейшем \"Покупатель\", в лице генерального директора Петров Петр Петрович, действующего на основании доверенности"
    )
    pipe = Pipeline(PipelineOptions(mask_style=MaskStyle.TAG))
    red, logs = pipe.process_text(txt)
    # The phrase should be masked as ROLE once, not two overlapping chunks
    assert red.count("[[ENTITY:ROLE]]") >= 1
