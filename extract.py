"""Load XL-Sum Hindi, pick the articles, and build a fact checklist for each -> data/checklists.jsonl."""

import json
import random
import re

from datasets import load_dataset
from indicnlp.normalize.indic_normalize import DevanagariNormalizer

import config

_normalizer = DevanagariNormalizer()

# One digit, Western (0-9) or Devanagari (०-९).
D = "[0-9०-९]"
MONTHS = [
    "जनवरी", "फ़रवरी", "फरवरी", "मार्च", "अप्रैल", "अप्रेल", "मई", "जून", "जुलाई",
    "अगस्त", "सितंबर", "सितम्बर", "अक्टूबर", "अक्तूबर", "नवंबर", "नवम्बर", "दिसंबर", "दिसम्बर",
]


def normalize_text(text):
    """Canonicalise Unicode (e.g. nukta forms of ज़/फ़) and collapse whitespace."""
    return re.sub(r"\s+", " ", _normalizer.normalize(text)).strip()


def normalize_digits(text):
    """Turn Devanagari digits into Western ones, so १२ and 12 compare equal."""
    return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))


def normalize_number(text):
    """Canonical form of a number: Western digits, no thousands commas ('1,50,000' -> '150000')."""
    return normalize_digits(text).replace(",", "")


# Month names go through the same normaliser as the articles, so nukta forms match.
_MONTHS = "|".join(sorted({normalize_text(m) for m in MONTHS}, key=len, reverse=True))
DATE_RE = re.compile(
    rf"{D}{{1,2}}[/.-]{D}{{1,2}}[/.-]{D}{{2,4}}"            # 12/05/2019, 12-05-19
    rf"|{D}{{1,2}}\s+(?:{_MONTHS})(?:\s*,?\s*{D}{{4}})?"     # 3 नवंबर, 3 नवंबर 2008
    rf"|(?:{_MONTHS})\s*,?\s*{D}{{4}}"                       # नवंबर 2008
)
# A run of digits, optionally with commas or a decimal point between digits.
# "2008." at a sentence end stays "2008" because a digit must follow the point.
NUMBER_RE = re.compile(rf"{D}+(?:[.,]{D}+)*")
# Sentence ends: Hindi danda or Western punctuation followed by a space.
SENTENCE_RE = re.compile(r".+?(?:[.।?!](?=\s)|$)", re.S)


def find_dates(text):
    """Return (start, end) spans of dates in the text."""
    return [m.span() for m in DATE_RE.finditer(text)]


def find_numbers(text, date_spans=()):
    """Return (start, end) spans of numbers that are not part of a date."""
    spans = []
    for m in NUMBER_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in date_spans):
            spans.append(m.span())
    return spans


def load_ner():
    """Load the Hindi NER model (needs a Hugging Face login, see README)."""
    from transformers import pipeline

    # No aggregation here: the pipeline's own word grouping breaks on this model,
    # whose uncased tokenizer strips Devanagari vowel signs. We group tokens into words ourselves.
    return pipeline("token-classification", model=config.NER_MODEL, aggregation_strategy="none")


def word_labels(sentence, tokens):
    """Give each whitespace word the label and score of its first sub-word token."""
    words = []
    for m in re.finditer(r"\S+", sentence):
        first = next((t for t in tokens if m.start() <= t["start"] < m.end()), None)
        label, score = (first["entity"], float(first["score"])) if first else ("O", 1.0)
        words.append({"start": m.start(), "end": m.end(), "label": label, "score": score,
                      "ends_in_punct": bool(re.search(r"[.,।?!)\"':;]$", m.group()))})
    return words


def merge_words(words):
    """Join consecutive words into entity spans using the B-/I- tags; return (start, end, type, score)."""
    spans, prev = [], None
    for word in words:
        if word["label"] == "O":
            prev = None
            continue
        tag, kind = word["label"].split("-")
        # An I- word continues the previous entity of the same type, unless punctuation
        # ("दस्ते," or "मोदी.") ended it.
        if tag == "I" and prev and prev["kind"] == kind and not prev["closed"]:
            prev["end"] = word["end"]
            prev["scores"].append(word["score"])
        else:
            prev = {"start": word["start"], "end": word["end"], "kind": kind, "scores": [word["score"]]}
            spans.append(prev)
        prev["closed"] = word["ends_in_punct"]
    return [(s["start"], s["end"], s["kind"], sum(s["scores"]) / len(s["scores"])) for s in spans]


def trim(text, start, end):
    """Shrink a span so it doesn't start or end with punctuation such as brackets or full stops."""
    while start < end and not re.match(r"[\wऀ-ॿ]", text[start]):
        start += 1
    while end > start and not re.match(r"[\wऀ-ॿ]", text[end - 1]):
        end -= 1
    return start, end


def find_entities(text, ner):
    """Return (start, end) spans of people, places and organisations, running NER one sentence at a time."""
    # Whole articles can exceed the model's 512-token limit, sentences never do.
    sentences = [m for m in SENTENCE_RE.finditer(text) if m.group().strip()]
    spans = []
    for sent, tokens in zip(sentences, ner([m.group() for m in sentences])):
        for start, end, kind, score in merge_words(word_labels(sent.group(), tokens)):
            start, end = trim(text, sent.start() + start, sent.start() + end)
            if (kind in config.NER_LABELS and score >= config.NER_MIN_SCORE and end - start >= 2
                    and not set(text[start:end].split()) <= config.ENTITY_STOPWORDS):
                spans.append((start, end))
    return spans


def unique(items):
    """Remove duplicates but keep the first-seen order."""
    return list(dict.fromkeys(items))


def build_checklist(text, ner):
    """Build the fact checklist for one article, plus the character spans the mixer must not touch."""
    date_spans = find_dates(text)
    number_spans = find_numbers(text, date_spans)
    entity_spans = find_entities(text, ner)
    checklist = {
        "entities": unique(text[s:e].strip() for s, e in entity_spans),
        "numbers": unique(normalize_number(text[s:e]) for s, e in number_spans),
        "dates": unique(normalize_digits(text[s:e]) for s, e in date_spans),
    }
    protected = sorted(set(date_spans + number_spans + entity_spans))
    return checklist, protected


def load_articles():
    """Load the Hindi test split, keep articles of MIN_WORDS..MAX_WORDS words, and draw N_ARTICLES with a fixed seed."""
    ds = load_dataset(
        config.DATASET_NAME,
        revision=config.DATASET_REVISION,
        data_files={"test": config.DATASET_FILE},
        split="test",
    )
    rows = []
    for row in ds:
        text = normalize_text(row["text"])
        n_words = len(text.split())
        if config.MIN_WORDS <= n_words <= config.MAX_WORDS:
            rows.append({"id": row["id"], "title": row["title"], "text": text,
                         "reference_summary": row["summary"], "n_words": n_words})
    random.Random(config.SEED).shuffle(rows)
    return rows[: config.N_ARTICLES]


def main():
    """Pick the articles, build every checklist, and write them to disk."""
    config.DATA_DIR.mkdir(exist_ok=True)
    articles = load_articles()
    config.ARTICLE_IDS_PATH.write_text("\n".join(a["id"] for a in articles) + "\n")
    ner = load_ner()
    with open(config.CHECKLISTS_PATH, "w", encoding="utf-8") as f:
        for i, article in enumerate(articles, 1):
            article["checklist"], article["protected_spans"] = build_checklist(article["text"], ner)
            f.write(json.dumps(article, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(articles)}] {article['id']}: "
                  + ", ".join(f"{len(v)} {k}" for k, v in article["checklist"].items()))
    print(f"Wrote {config.CHECKLISTS_PATH}")


if __name__ == "__main__":
    main()
