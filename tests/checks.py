"""Minimal sanity checks. Run from the project root after extract.py and mix.py: python tests/checks.py"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from extract import find_dates, find_numbers, normalize_digits, normalize_number, normalize_text


def spans_text(text, spans):
    """The substrings covered by a list of spans."""
    return [text[s:e] for s, e in spans]


# --- Extractor: numbers and dates in hand-written sentences ---
s1 = normalize_text("सरकार ने 15 लाख रुपये की मदद दी और 1,50,000 लोगों को फ़ायदा हुआ.")
assert [normalize_number(n) for n in spans_text(s1, find_numbers(s1))] == ["15", "150000"]

s2 = normalize_text("चुनाव 3 नवंबर 2008 को हुए और नतीजे 12/05/2009 को आए.")
assert spans_text(s2, find_dates(s2)) == ["3 नवंबर 2008", "12/05/2009"]
# Digits inside a date are not counted again as separate numbers.
assert find_numbers(s2, find_dates(s2)) == []

s3 = normalize_text("फ़रवरी 2019 में १२ लोग घायल हुए और दर 2.5 प्रतिशत रही.")
assert spans_text(s3, find_dates(s3)) == ["फ़रवरी 2019"]
assert [normalize_number(n) for n in spans_text(s3, find_numbers(s3, find_dates(s3)))] == ["12", "2.5"]

# --- Devanagari and Western digits normalise to the same value ---
assert normalize_digits("१२३४५६७८९०") == "1234567890"
assert normalize_number("१,५०,०००") == normalize_number("1,50,000") == "150000"

# --- Mixer checks on the real output ---
articles = {a["id"]: a for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
rows = [json.loads(line) for line in open(config.MIXED_PATH, encoding="utf-8")]
assert len(rows) == len(articles) * len(config.LEVELS)

for row in rows:
    article = articles[row["id"]]
    # Replaced share is within +/-3 points of the target level. Rounding to whole words
    # can be off by up to 50/n points, so articles with < 17 eligible words are exempt.
    if row["n_eligible"]:
        assert abs(100 * row["n_replaced"] / row["n_eligible"] - row["level"]) <= 3 or row["n_eligible"] < 17, row["id"]
    # Entities, numbers and dates of the original survive unchanged.
    for fact in spans_text(article["text"], article["protected_spans"]):
        assert fact in row["text"], (row["id"], row["level"], fact)

# Levels are nested: what is replaced at one level is also replaced at every higher level.
by_article = {}
for row in rows:
    by_article.setdefault(row["id"], {})[row["level"]] = row["replaced"]
for levels in by_article.values():
    for low, high in zip(config.LEVELS, config.LEVELS[1:]):
        remaining = list(levels[high])
        for word in levels[low]:
            remaining.remove(word)  # raises ValueError if a lower-level word is missing

# Level 0 is the untouched original.
assert all(row["text"] == articles[row["id"]]["text"] for row in rows if row["level"] == 0)

# --- Scorer: recall 1.0 for a summary containing the full checklist, 0.0 for an empty one ---
from score import contains, first_sentences, match_key, recall

checklist = {"entities": ["नरेंद्र मोदी", "दिल्ली"], "numbers": ["150000", "12"], "dates": ["3 नवंबर 2008"]}
full = "नरेंद्र मोदी 3 नवंबर 2008 को दिल्ली गए, जहाँ 1,50,000 लोग और १२ मंत्री थे."
assert recall(full, checklist)[0] == 1.0
assert recall("", checklist)[0] == 0.0
# Whole-word matching: the entity भारत is not found inside भारतीय, and 13 is not found inside 130.
assert not contains(match_key("भारतीय टीम जीती."), "भारत")
assert recall("करीब 130 लोग.", {"entities": [], "numbers": ["13"], "dates": []})[0] == 0.0
# Truncation keeps the first 3 sentences; a decimal point does not end a sentence.
assert first_sentences("एक. दर 2.5 रही. तीन।\nचार. पाँच.") == "एक. दर 2.5 रही. तीन।"
# A lead-in line ending in a colon ("here is the summary:") is not part of the summary.
assert first_sentences("यहाँ 2-3 वाक्यों का सार है:\nएक. दो. तीन. चार.") == "एक. दो. तीन."



# --- Stretch (targeted vs random mixing) ---
from mix import same_name
from score import english_form, recall_either_form

# A fact word is replaced only by the Latin spelling of the same name, never by a translation.
assert same_name("पाकिस्तान", "pakistan") and same_name("कोर्ट", "court")
assert not same_name("मुलायम", "soft") and not same_name("सिंह", "leo")
# Recall accepts a fact in either its Hindi or its English form.
fe = {"नरेंद्र": "Narendra", "मोदी": "Modi"}
assert english_form("नरेंद्र मोदी", fe) == "Narendra Modi"
assert recall_either_form("Narendra Modi आए.", {"entities": ["नरेंद्र मोदी"], "numbers": [], "dates": []}, fe)[0] == 1.0
if config.STRETCH_MIXED_PATH.exists():
    stretch = {(r["id"], r["level"]): r for r in map(json.loads, open(config.STRETCH_MIXED_PATH, encoding="utf-8"))}
    for article_id, article in articles.items():
        targeted, rand = stretch[(article_id, "targeted")], stretch[(article_id, "random")]
        # Both conditions replace the same number of words; random never touches a fact.
        assert targeted["n_replaced"] == rand["n_replaced"], article_id
        for fact in spans_text(article["text"], article["protected_spans"]):
            assert fact in rand["text"], (article_id, fact)
print("All checks passed.")
