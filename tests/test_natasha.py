# Пример (идея использования):
import os
from redactor.natasha_ner import NatashaNER
from redactor.patterns    import regex_find_spans
from redactor.presidio_recognizers import build_analyzer, analyze_text

ner = NatashaNER()

_flag  = os.getenv("REDACTOR_NATASHA_FULL", "0").lower()
print(f"REDACTOR_NATASHA_FULL={_flag}")

with open('D:/CODE/PY/NOIKEY/tests/documents/contract.txt', 'r', encoding='utf-8') as f:
    content = f.read()

text   = content
#print("Text snippet:", text)

_spans = ner.analyze(text)
print("Spans (light mode):")
for span in _spans:
    print(f"    {span.label}: '{span.start}-{span.end}' score={span.score} text='{text[span.start:span.end]}'")

_spans_regex = regex_find_spans(text)
print("Spans (regex):")
for span in _spans_regex:
    print(f"    {span.label}: '{span.start}-{span.end}' score={span.score} text='{text[span.start:span.end]}'")

_spans_presidio = analyze_text(build_analyzer(), text, min_score=0)
print("Spans (presidio):")
for span in _spans_presidio:
    print(f"    {span.label}: '{span.start}-{span.end}' score={span.score} text='{text[span.start:span.end]}'")
