import yaml, re, os

# путь поправь под фактическое расположение файла:
cfg_path = os.path.join("config", "patterns.yaml")  # либо просто "patterns.yaml"
with open(cfg_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

pattern_text = cfg["regex"]["org_pattern"]
print(pattern_text)  # тут увидишь многострочный шаблон

org_re = re.compile(pattern_text, flags=re.IGNORECASE | re.UNICODE | re.VERBOSE)

texts = ['Общество с ограниченной — ответственностью — «Современные — Технологии — Газовых — Турбин»',
         'публичное акционерное общество «Органический синтез»']
for text in texts:
    print(bool(org_re.search(text)))  # теперь должно быть True
