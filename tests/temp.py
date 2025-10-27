import os

_dpath = os.path.join(os.getcwd(), "config", "domain.txt")

if os.path.exists(_dpath):
    with open(_dpath, "r", encoding="utf-8") as f:
        DOMAIN_SPECIFIC_WORDS = {line.strip() for line in f if line.strip()}

print(DOMAIN_SPECIFIC_WORDS)