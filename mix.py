"""Make code-mixed versions of every article at each mixing level -> data/mixed.jsonl."""

import difflib
import json
import random
import re
import sys
import urllib.request

import stanza
from wordfreq import zipf_frequency

import config
from extract import normalize_text

DEVANAGARI_WORD = re.compile(r"^[ऀ-ॿ]+$")
ENGLISH_WORD = re.compile(r"^[a-z]+$")


def load_dictionary():
    """Map each Hindi word to its single most common English translation from the MUSE word list."""
    if not config.DICT_PATH.exists():
        config.DICT_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.DICT_URL, config.DICT_PATH)
    candidates = {}
    for line in config.DICT_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and ENGLISH_WORD.match(parts[1]):
            candidates.setdefault(normalize_text(parts[0]), []).append(parts[1])
    dictionary = {}
    for hindi, english in candidates.items():
        best = max(english, key=lambda w: zipf_frequency(w, "en"))
        if zipf_frequency(best, "en") >= config.MIN_ZIPF:
            dictionary[hindi] = best
    return dictionary


def load_tagger():
    """Load Stanza's Hindi tokenizer, part-of-speech tagger and lemmatizer."""
    stanza.download(config.STANZA_LANG, processors="tokenize,pos,lemma", verbose=False)
    return stanza.Pipeline(config.STANZA_LANG, processors="tokenize,pos,lemma", verbose=False)


def tag_words(text, tagger):
    """Return every word of the text with its POS tag, lemma and character span."""
    words = []
    for sentence in tagger(text).sentences:
        for token in sentence.tokens:
            word = token.words[0]  # Hindi has no multi-word tokens, so one word per token
            words.append({"text": token.text, "upos": word.upos, "lemma": word.lemma or token.text,
                          "plural": "Number=Plur" in (word.feats or ""),
                          "start": token.start_char, "end": token.end_char})
    return words


def strip_nukta(text):
    """Remove the nukta dot (फ़ -> फ). BBC Hindi writes फ़ैसला, the dictionary often has फैसला."""
    return text.replace("़", "")


def translate(word, dictionary):
    """English equivalent of a word, trying its forms in order, with and without nukta. None if unknown."""
    # Singular फ़ैसले ("इस फ़ैसले के बाद") is spelled like the plural, and the dictionary maps
    # that spelling to "decisions". So singular words are looked up by lemma first
    # (फ़ैसला -> decision) and plural words by their exact form first (किसानों -> farmers).
    forms = [word["text"], word["lemma"]] if word["plural"] else [word["lemma"], word["text"]]
    for form in forms + [strip_nukta(f) for f in forms]:
        if form in dictionary:
            return dictionary[form]
    return None


def is_eligible(word, protected_spans):
    """True for a content word (noun or adjective) that is not part of a fact."""
    if word["upos"] not in config.CONTENT_POS:
        return False
    if not DEVANAGARI_WORD.match(word["text"]):
        return False
    return not any(s < word["end"] and word["start"] < e for s, e in protected_spans)


def eligible_words(text, protected_spans, tagger, dictionary):
    """List the words that may be replaced, each with its English equivalent."""
    eligible = []
    for word in tag_words(text, tagger):
        if is_eligible(word, protected_spans):
            english = translate(word, dictionary)
            # Words the dictionary can't translate are left out of the eligible set,
            # so "L%" is always L% of words we are actually able to replace.
            if english:
                eligible.append({**word, "english": english})
    return eligible


def replacement_order(n, article_id):
    """A fixed random order of the eligible words, seeded by article id."""
    order = list(range(n))
    random.Random(f"{config.SEED}-{article_id}").shuffle(order)
    return order


def mix_text(text, eligible, chosen):
    """Replace the chosen eligible words with English, keeping everything else exactly as it was."""
    pieces, last = [], 0
    for i in sorted(chosen, key=lambda i: eligible[i]["start"]):
        word = eligible[i]
        pieces += [text[last:word["start"]], word["english"]]
        last = word["end"]
    return "".join(pieces) + text[last:]


def mix_article(article, tagger, dictionary):
    """Return one row per mixing level for an article."""
    eligible = eligible_words(article["text"], article["protected_spans"], tagger, dictionary)
    order = replacement_order(len(eligible), article["id"])
    rows = []
    for level in config.LEVELS:
        # Taking a prefix of one fixed order makes the levels nested:
        # every word replaced at 20% is also replaced at 40%, and so on.
        chosen = order[: round(level / 100 * len(eligible))]
        rows.append({
            "id": article["id"],
            "level": level,
            "text": mix_text(article["text"], eligible, chosen),
            "n_eligible": len(eligible),
            "n_replaced": len(chosen),
            "replaced": sorted(eligible[i]["text"] for i in chosen),
        })
    return rows, {w["text"]: w["english"] for w in eligible}


def write_review_samples(articles, rows, n_articles=4, n_words=70):
    """Write the opening of a few articles at every level to a text file for a human to check."""
    picked = random.Random(config.SEED).sample(articles, n_articles)
    lines = []
    for article in picked:
        lines += [f"=== {article['id']} | {article['title']}",
                  f"checklist: {json.dumps(article['checklist'], ensure_ascii=False)}"]
        for row in rows:
            if row["id"] == article["id"]:
                opening = " ".join(row["text"].split()[:n_words])
                lines += [f"--- level {row['level']}% ({row['n_replaced']}/{row['n_eligible']} words)", opening]
        lines.append("")
    config.SAMPLES_PATH.write_text("\n".join(lines), encoding="utf-8")


# --- Stretch: targeted vs random mixing ---

# A rough Devanagari -> Latin spelling, only used to ask "is this English word the same name?".
CONSONANTS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
                      "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h".split()))
VOWELS = dict(zip("अआइईउऊऋएऐओऔऑ", "a a i i u u ri e ai o au o".split()))
VOWEL_SIGNS = dict(zip("ािीुूृेैोौॉ", "a i i u u ri e ai o au o".split()))


def romanise(word):
    """Rough Latin spelling of a Devanagari word, e.g. पाकिस्तान -> pakistan."""
    out = []
    for i, ch in enumerate(word):
        nxt = word[i + 1] if i + 1 < len(word) else ""
        if ch in CONSONANTS:
            # A consonant carries an "a" unless a vowel sign, virama or nukta follows,
            # or it ends the word (Hindi drops that final "a": भारत = bharat).
            out.append(CONSONANTS[ch] + ("" if not nxt or nxt in VOWEL_SIGNS or nxt in "़्" else "a"))
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[ch])
        elif ch in "ंँ":
            out.append("n")
    return "".join(out)


def same_name(hindi, english):
    """True if the English word is the Latin spelling of the Hindi one (कोर्ट/court), not a translation (मुलायम/soft)."""
    return difflib.SequenceMatcher(None, romanise(strip_nukta(hindi)), english).ratio() >= config.STRETCH_NAME_SIMILARITY


def fact_words(text, protected_spans, tagger, dictionary):
    """Words inside facts that have an English form of the same name, each with that form capitalised."""
    found = []
    for word in tag_words(text, tagger):
        inside = any(s < word["end"] and word["start"] < e for s, e in protected_spans)
        if inside and DEVANAGARI_WORD.match(word["text"]):
            english = translate(word, dictionary)
            if english and same_name(word["text"], english):
                # Names are capitalised in Hinglish; a word with no vowels is an acronym (bjp -> BJP).
                english = english.upper() if not re.search(r"[aeiou]", english) else english.capitalize()
                found.append({**word, "english": english})
    return found


def stretch_rows(article, tagger, dictionary):
    """One targeted and one random row for an article, replacing the same number of words."""
    eligible = eligible_words(article["text"], article["protected_spans"], tagger, dictionary)
    facts = fact_words(article["text"], article["protected_spans"], tagger, dictionary)
    k = min(len(facts), len(eligible))  # equal counts in both conditions
    fact_order = list(range(len(facts)))
    random.Random(f"{config.SEED}-{article['id']}-facts").shuffle(fact_order)
    chosen = {
        # Same prefix of the same shuffle as the main experiment, so random is nested with its levels.
        "random": (eligible, replacement_order(len(eligible), article["id"])[:k]),
        "targeted": (facts, fact_order[:k]),
    }
    # Every fact word's English form, used by the scorer to accept either spelling of a fact.
    fact_english = {w["text"]: w["english"] for w in facts}
    return [{
        "id": article["id"],
        "level": condition,  # generate.py keys rows by "level"; here it holds the condition name
        "text": mix_text(article["text"], words, picked),
        "n_replaced": len(picked),
        "replaced": sorted(words[i]["text"] for i in picked),
        "fact_english": fact_english,
    } for condition, (words, picked) in chosen.items()]


def main_stretch():
    """Write the targeted and random versions of every article -> data/stretch_mixed.jsonl."""
    articles = [json.loads(line) for line in open(config.CHECKLISTS_PATH, encoding="utf-8")]
    tagger, dictionary = load_tagger(), load_dictionary()
    with open(config.STRETCH_MIXED_PATH, "w", encoding="utf-8") as f:
        for i, article in enumerate(articles, 1):
            rows = stretch_rows(article, tagger, dictionary)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(articles)}] {article['id']}: {rows[0]['n_replaced']} words per condition")
    print(f"Wrote {config.STRETCH_MIXED_PATH}")


def main():
    """Mix every article at every level and save the rows and the translations used."""
    if "--stretch" in sys.argv:
        return main_stretch()
    articles = [json.loads(line) for line in open(config.CHECKLISTS_PATH, encoding="utf-8")]
    tagger, dictionary = load_tagger(), load_dictionary()
    all_rows, translations = [], {}
    for i, article in enumerate(articles, 1):
        rows, used = mix_article(article, tagger, dictionary)
        all_rows += rows
        translations.update(used)
        print(f"[{i}/{len(articles)}] {article['id']}: {rows[0]['n_eligible']} eligible words")
    with open(config.MIXED_PATH, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    # Every Hindi -> English choice we made, saved so it can be inspected or reused.
    config.TRANSLATIONS_PATH.write_text(
        json.dumps(dict(sorted(translations.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    write_review_samples(articles, all_rows)
    print(f"Wrote {config.MIXED_PATH} ({len(all_rows)} rows) and {config.SAMPLES_PATH}")


if __name__ == "__main__":
    main()
