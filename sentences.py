"""Sentence splitting shared by extraction, scoring and the two-pass `verified` prompt.

It has no dependencies beyond `re` and `config`, so `generate.py` can use it on Kaggle
to cut a draft summary exactly the way `score.py` cuts summaries before scoring.
"""

import re

import config

# Sentence ends: Hindi danda or Western punctuation followed by a space.
SENTENCE_RE = re.compile(r".+?(?:[.।?!](?=\s)|$)", re.S)


def first_sentences(text, n=config.SCORE_SENTENCES):
    """Keep the first n sentences of a summary (line breaks also end a sentence)."""
    lines = text.strip().split("\n")
    # 25 summaries open with a lead-in such as "हिंदी में इस खबर का संक्षेप निम्नलिखित है:"
    # ("here is the summary:"). It is not part of the summary, so it is dropped.
    if lines[0].strip().endswith(":"):
        lines = lines[1:]
    sentences = []
    for line in lines:
        sentences += [m.group().strip() for m in SENTENCE_RE.finditer(line) if m.group().strip()]
    return " ".join(sentences[:n])
