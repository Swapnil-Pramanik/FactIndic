"""Score every summary against its article's checklist -> results/scores.jsonl, metrics.csv, chart.png."""

import argparse
import json
import random
import re

import matplotlib

matplotlib.use("Agg")  # write the PNG without needing a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from extract import (SENTENCE_RE, find_dates, find_entities, find_numbers, load_ner,
                     normalize_digits, normalize_number, normalize_text)


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


def match_key(text):
    """Normalise text for fact matching: canonical Unicode, Western digits, no nukta, no punctuation."""
    text = normalize_digits(normalize_text(text)).replace("़", "").lower()
    # The danda (।) sits inside the Devanagari block, so it is removed explicitly.
    text = re.sub(r"[^\wऀ-ॿ]|[।॥]", " ", text)
    # Spaces on both ends let us match whole words only: " भारत " is not found in " भारतीय ".
    return " " + " ".join(text.split()) + " "


def contains(haystack_key, fact):
    """True if the fact appears as whole words in text already passed through match_key."""
    return match_key(fact) in haystack_key


def numbers_in(text, skip_dates=True):
    """Set of normalised numbers in a text (by default leaving out digits inside dates)."""
    dates = find_dates(text) if skip_dates else []
    return {normalize_number(text[s:e]) for s, e in find_numbers(text, dates)}


def recall(summary, checklist):
    """Share of checklist facts found in the summary, overall and per fact type."""
    key, nums = match_key(summary), numbers_in(summary, skip_dates=False)
    hits = {
        "entities": [f for f in checklist["entities"] if contains(key, f)],
        "numbers": [f for f in checklist["numbers"] if f in nums],
        "dates": [f for f in checklist["dates"] if contains(key, f)],
    }
    total = sum(len(v) for v in checklist.values())
    per_type = {k: len(hits[k]) / len(checklist[k]) if checklist[k] else None for k in hits}
    overall = sum(len(v) for v in hits.values()) / total if total else None
    return overall, per_type


def precision(summary, article, ner):
    """Facts in the summary, which of them are not in the original article, and the share that are."""
    key = match_key(article)
    article_numbers = numbers_in(article, skip_dates=False)  # a year inside a date still counts
    date_spans = find_dates(summary)
    found = (
        [("entity", summary[s:e]) for s, e in find_entities(summary, ner)]
        + [("number", n) for n in sorted(numbers_in(summary))]  # sorted: sets have no fixed order
        + [("date", normalize_digits(summary[s:e])) for s, e in date_spans]
    )
    unsupported = [
        (kind, fact) for kind, fact in found
        if (fact not in article_numbers if kind == "number" else not contains(key, fact))
    ]
    value = 1 - len(unsupported) / len(found) if found else None
    return value, found, unsupported


def without_latin(found, unsupported):
    """Precision again, leaving out "entities" that contain English letters.

    These are mostly English words the model copied from the mixed input that NER then
    tagged ("अमरीकी embassy", "plea"). They are not invented facts, and they only appear
    once the input is mixed, so they pull plain precision down as mixing rises.
    Reported next to `precision`, never instead of it.
    """
    def keep(fact):
        return not (fact[0] == "entity" and re.search(r"[A-Za-z]", fact[1]))
    found, unsupported = [f for f in found if keep(f)], [f for f in unsupported if keep(f)]
    return 1 - len(unsupported) / len(found) if found else None


def score_all(generations_path=config.GENERATIONS_PATH):
    """Score every generated summary; return one dict per summary."""
    articles = {a["id"]: a for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    mixed = {(m["id"], m["level"]): m["text"] for m in map(json.loads, open(config.MIXED_PATH, encoding="utf-8"))}
    ner = load_ner()
    rows = []
    for g in map(json.loads, open(generations_path, encoding="utf-8")):
        article = articles[g["id"]]
        summary = first_sentences(g["summary"])
        rec, rec_type = recall(summary, article["checklist"])
        prec, found, unsupported = precision(summary, article["text"], ner)
        input_words = mixed[(g["id"], g["level"])].split()
        rows.append({
            "id": g["id"], "level": g["level"], "prompt": g["prompt"],
            "scored_summary": summary,
            "recall": rec, **{f"recall_{k}": v for k, v in rec_type.items()},
            "precision": prec,
            "precision_no_latin": without_latin(found, unsupported),
            "n_facts_found": len(found), "n_unsupported": len(unsupported),
            "unsupported": [f"{kind}: {fact}" for kind, fact in unsupported],
            "summary_words": len(summary.split()),
            # Share of summary words containing English letters (does the model answer in English?).
            "summary_latin_share": sum(bool(re.search(r"[A-Za-z]", w)) for w in summary.split()) / max(len(summary.split()), 1),
            "compression": len(summary.split()) / article["n_words"],
            # Share of the model's input that was English, to read the level labels by.
            "input_english_share": sum(bool(re.fullmatch(r"[a-z]+", w)) for w in input_words) / len(input_words),
            "output_tokens": g["n_output_tokens"], "hit_max_tokens": g["hit_max_tokens"], "time_s": g["time_s"],
        })
    return rows


def add_f1(df):
    """Per-summary fact F1 = harmonic mean of checklist recall and fact precision.
    Undefined (NaN) when precision is undefined, i.e. the summary contains no facts at all."""
    r, p = df["recall"], df["precision"].astype(float)
    df["f1"] = np.where(r + p > 0, 2 * r * p / (r + p), 0.0)
    df.loc[p.isna(), "f1"] = np.nan
    return df


def metrics_table(scores):
    """Average each measure per (level, prompt). Means skip summaries where a measure is undefined."""
    df = add_f1(pd.DataFrame(scores))
    # An ordered category makes the table list "plain" before "checklist" within each level.
    df["prompt"] = pd.Categorical(df["prompt"], categories=config.PROMPTS, ordered=True)
    table = df.groupby(["level", "prompt"], observed=True).agg(
        n=("id", "size"),
        input_english_share=("input_english_share", "mean"),
        recall=("recall", "mean"),
        recall_entities=("recall_entities", "mean"),
        recall_numbers=("recall_numbers", "mean"),  # only the 66 articles that have numbers
        precision=("precision", "mean"),            # only summaries containing at least one fact
        precision_no_latin=("precision_no_latin", "mean"),
        f1=("f1", "mean"),                          # mean of per-summary F1
        n_without_facts=("precision", lambda s: int(s.isna().sum())),
        unsupported_per_summary=("n_unsupported", "mean"),
        summary_words=("summary_words", "mean"),
        summary_latin_share=("summary_latin_share", "mean"),
        compression=("compression", "mean"),
        output_tokens_raw=("output_tokens", "mean"),
        hit_max_tokens=("hit_max_tokens", "mean"),
        time_s=("time_s", "mean"),
    ).reset_index()
    return table.round(3)


def plot(table):
    """Recall and precision against mixing level, one line per prompt, in two panels with a shared 0-1 scale."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True, facecolor="#fcfcfb")
    for ax, metric in zip(axes, ["recall", "precision"]):
        ax.set_facecolor("#fcfcfb")
        ends = {}
        for prompt in config.PROMPTS:
            part = table[table["prompt"] == prompt]
            ax.plot(part["level"], part[metric], color=config.PROMPT_COLORS[prompt], linewidth=2,
                    marker="o", markersize=6, label=prompt)
            ends[prompt] = part[metric].iloc[-1]
        # Direct labels at the line ends, in text colour (the marker carries the identity).
        # When the two ends are close, push the higher label up and the lower one down.
        higher = max(ends, key=ends.get)
        for prompt, y in ends.items():
            nudge = 0 if abs(ends["plain"] - ends["checklist"]) > 0.05 else (6 if prompt == higher else -6)
            ax.annotate(prompt, (config.LEVELS[-1], y), xytext=(8, nudge), textcoords="offset points",
                        va="center", fontsize=9, color="#52514e")
        ax.set_title(metric.capitalize(), loc="left", color="#0b0b0b", fontsize=11)
        ax.set_xticks(config.LEVELS)
        ax.set_xlabel("Mixing level (% of eligible words in English)")
        ax.set_xlim(-5, 95)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#e8e7e3", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(f"Fact recall and precision of {config.MODEL_NAME.split('/')[-1]} summaries "
                 f"(first {config.SCORE_SENTENCES} sentences, 100 articles)", x=0.01, ha="left",
                 color="#0b0b0b", fontsize=11)
    fig.tight_layout()
    fig.savefig(config.CHART_PATH, dpi=200, facecolor=fig.get_facecolor())


def write_sheet(df, path):
    """Write a CSV for hand-filling, refusing to overwrite one that may already hold manual work."""
    if path.exists():
        raise SystemExit(f"{path} already exists; delete it yourself if you really want a fresh sheet.")
    # utf-8-sig adds a byte-order mark so Excel and Numbers show Devanagari correctly.
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_sheets():
    """Write the blind error-tagging sheet (+ its key) and the checklist-quality sheet."""
    articles = {a["id"]: a for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    scores = {(s["id"], s["level"], s["prompt"]): s for s in map(json.loads, open(config.SCORES_PATH, encoding="utf-8"))}
    rng = random.Random(config.SEED)

    # The same articles under every condition, so condition differences are not article differences.
    items = []
    for article_id in rng.sample(sorted(articles), config.TAG_ARTICLES):
        conditions = [(level, prompt) for level in config.TAG_LEVELS for prompt in config.PROMPTS]
        rng.shuffle(conditions)  # within an article, the order gives no hint of the condition
        for level, prompt in conditions:
            s = scores[(article_id, level, prompt)]
            items.append({"article_id": article_id, "level": level, "prompt": prompt,
                          "summary": s["scored_summary"], "auto_unsupported": "; ".join(s["unsupported"])})
    for i, item in enumerate(items, 1):
        item["item"] = i
    key = pd.DataFrame(items)[["item", "article_id", "level", "prompt"]]
    # Blind sheet: no level or prompt. English words in a summary can still hint at mixing; that's unavoidable.
    sheet = pd.DataFrame([{
        "item": it["item"], "article_id": it["article_id"],
        "original_article": articles[it["article_id"]]["text"],
        "summary": it["summary"],
        "auto_unsupported (hint, may be wrong)": it["auto_unsupported"],
        "tags": "", "notes": "",
    } for it in items])
    write_sheet(sheet, config.ERROR_TAGS_PATH)
    key.to_csv(config.ERROR_TAGS_KEY_PATH, index=False)

    # Checklist quality: 20 articles, the student counts wrong and missed facts.
    check_ids = random.Random(config.SEED + 1).sample(sorted(articles), config.CHECKLIST_CHECK_ARTICLES)
    check = pd.DataFrame([{
        "article_id": i, "original_article": articles[i]["text"],
        **{k: "; ".join(v) for k, v in articles[i]["checklist"].items()},
        "n_items": sum(len(v) for v in articles[i]["checklist"].values()),
        "n_wrong (not a real name/number/date)": "", "n_missed (in article, not in checklist)": "", "notes": "",
    } for i in check_ids])
    write_sheet(check, config.CHECKLIST_CHECK_PATH)
    print(f"Wrote {config.ERROR_TAGS_PATH} ({len(sheet)} summaries), {config.ERROR_TAGS_KEY_PATH}, "
          f"{config.CHECKLIST_CHECK_PATH} ({len(check)} articles)")


def summarise_tags():
    """Count the hand-filled error tags per (level, prompt) -> error_tag_counts.csv."""
    tags = pd.read_csv(config.ERROR_TAGS_PATH, encoding="utf-8-sig").merge(
        pd.read_csv(config.ERROR_TAGS_KEY_PATH), on=["item", "article_id"])
    rows = []
    for _, row in tags.iterrows():
        labels = [t.strip().lower() for t in str(row["tags"]).split(";") if t.strip() and str(row["tags"]) != "nan"]
        if not labels:
            raise SystemExit(f"item {row['item']} has no tag; use 'none' if the summary has no error")
        for label in labels:
            if label not in config.TAG_CATEGORIES:
                raise SystemExit(f"item {row['item']}: unknown tag '{label}'. Allowed: {config.TAG_CATEGORIES}")
            rows.append({"level": row["level"], "prompt": row["prompt"], "tag": label})
    df = pd.DataFrame(rows)
    # One row per (level, prompt), one column per tag; a summary can carry several tags.
    counts = pd.crosstab([df["level"], df["prompt"]], df["tag"]).reindex(columns=config.TAG_CATEGORIES, fill_value=0)
    counts.to_csv(config.ERROR_COUNTS_PATH)
    print(counts.to_string())
    print(f"Wrote {config.ERROR_COUNTS_PATH}")


def english_form(fact, fact_english):
    """The fact with each word that targeted mixing would replace swapped for its English form."""
    return " ".join(fact_english.get(word, word) for word in fact.split())


def recall_either_form(summary, checklist, fact_english):
    """Recall where a fact counts in either its Hindi form or its English form (मोदी or Modi)."""
    key, nums = match_key(summary), numbers_in(summary, skip_dates=False)
    def found(fact):
        return contains(key, fact) or contains(key, english_form(fact, fact_english))
    entity_hits = sum(found(f) for f in checklist["entities"])
    hits = entity_hits + sum(found(f) for f in checklist["dates"]) + sum(f in nums for f in checklist["numbers"])
    total = sum(len(v) for v in checklist.values())
    return hits / total, entity_hits / len(checklist["entities"]) if checklist["entities"] else None


def score_stretch(paths=None):
    """Score the targeted vs random summaries -> stretch_scores.jsonl and stretch_metrics.csv."""
    paths = paths or config.model_paths("1b")
    articles = {a["id"]: a for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    mixed = {(m["id"], m["level"]): m for m in map(json.loads, open(config.STRETCH_MIXED_PATH, encoding="utf-8"))}
    ner = load_ner()
    rows = []
    for g in map(json.loads, open(paths["stretch_generations"], encoding="utf-8")):
        article, row = articles[g["id"]], mixed[(g["id"], g["level"])]
        summary = first_sentences(g["summary"])
        rec, rec_entities = recall_either_form(summary, article["checklist"], row["fact_english"])
        # A fact is supported if it is in the original article in either spelling, so both
        # conditions are checked against the original plus its fully targeted (English-name) version.
        support = article["text"] + "\n" + mixed[(g["id"], "targeted")]["text"]
        prec, found, unsupported = precision(summary, support, ner)
        rows.append({
            "id": g["id"], "condition": g["level"], "prompt": g["prompt"], "scored_summary": summary,
            "n_replaced": row["n_replaced"], "recall": rec, "recall_entities": rec_entities, "precision": prec,
            "n_unsupported": len(unsupported), "unsupported": [f"{kind}: {fact}" for kind, fact in unsupported],
            "summary_words": len(summary.split()),
            "summary_latin_share": sum(bool(re.search(r"[A-Za-z]", w)) for w in summary.split()) / max(len(summary.split()), 1),
            "hit_max_tokens": g["hit_max_tokens"],
        })
    with open(paths["stretch_scores"], "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    df = add_f1(pd.DataFrame(rows))
    df["prompt"] = pd.Categorical(df["prompt"], categories=config.PROMPTS, ordered=True)
    table = df.groupby(["condition", "prompt"], observed=True).agg(
        n=("id", "size"), words_replaced=("n_replaced", "mean"),
        recall=("recall", "mean"), recall_entities=("recall_entities", "mean"),
        precision=("precision", "mean"), f1=("f1", "mean"), unsupported_per_summary=("n_unsupported", "mean"),
        summary_words=("summary_words", "mean"), summary_latin_share=("summary_latin_share", "mean"),
        hit_max_tokens=("hit_max_tokens", "mean"),
    ).reset_index().round(3)
    table.to_csv(paths["stretch_metrics"], index=False)
    print(table.to_string(index=False))
    print(f"Wrote {paths['stretch_scores']}, {paths['stretch_metrics']}")


def paired_diffs(df, measure, column="condition", a="targeted", b="random", n_samples=config.BOOTSTRAP_SAMPLES):
    """Mean a-minus-b difference per article, with a 95% bootstrap interval over articles."""
    wide = df.pivot(index="id", columns=column, values=measure).dropna()
    diffs = (wide[a] - wide[b]).to_numpy()
    rng = np.random.default_rng(config.SEED)
    means = np.sort(rng.choice(diffs, size=(n_samples, len(diffs))).mean(axis=1))
    return {"n": len(diffs), "diff": diffs.mean(), "low": np.quantile(means, 0.025), "high": np.quantile(means, 0.975)}


def plot_sizes(table):
    """Small multiples: recall (top) and precision (bottom) against mixing level, one column per model."""
    sizes = list(config.MODELS)
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.7 * len(sizes), 6), sharey="row", sharex=True,
                             facecolor="white")
    for col, tag in enumerate(sizes):
        for row, measure in enumerate(["recall", "precision"]):
            ax = axes[row, col]
            for prompt in config.PROMPTS:
                part = table[(table["model"] == tag) & (table["prompt"] == prompt)]
                ax.plot(part["level"], part[measure], marker="o", color=config.PROMPT_COLORS[prompt], label=prompt)
            ax.set_xticks(config.SCALE_LEVELS)
            ax.grid(axis="y", color="#e5e5e5")
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(config.MODELS[tag].split("/")[-1].replace("-Instruct", "").replace("-it", "").replace("gemma", "Gemma"), fontsize=11)
            else:
                ax.set_xlabel("mixing level (%)")
            if col == 0:
                ax.set_ylabel(measure)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Fact recall and precision vs mixing level, by model")
    fig.tight_layout()
    fig.savefig(config.SIZE_CHART_PATH, dpi=200, facecolor=fig.get_facecolor())


def compare_sizes():
    """Combine every model's scores -> size_metrics.csv, size_stretch_metrics.csv, size_stretch_diffs.csv, chart."""
    main_tables, stretch_tables, diffs, mixing = [], [], [], []
    for tag in config.MODELS:
        paths = config.model_paths(tag)
        scores = pd.DataFrame(map(json.loads, open(paths["scores"], encoding="utf-8")))
        scores = scores[scores["level"].isin(config.SCALE_LEVELS)]
        main_tables.append(metrics_table(scores.to_dict("records")).assign(model=tag))
        # Effect of mixing: level 80 minus level 0 on the same article, and checklist minus plain.
        for prompt in config.PROMPTS:
            for measure in ["recall", "precision", "n_unsupported"]:
                mixing.append({"model": tag, "comparison": f"L80-L0 {prompt}", "measure": measure,
                               **paired_diffs(scores[scores["prompt"] == prompt], measure, "level", 80, 0)})
        for level in config.SCALE_LEVELS:
            for measure in ["recall", "precision"]:
                mixing.append({"model": tag, "comparison": f"checklist-plain L{level}", "measure": measure,
                               **paired_diffs(scores[scores["level"] == level], measure, "prompt", "checklist", "plain")})
        stretch_tables.append(pd.read_csv(paths["stretch_metrics"]).assign(model=tag))
        stretch = pd.DataFrame(map(json.loads, open(paths["stretch_scores"], encoding="utf-8")))
        for prompt in config.PROMPTS:
            for measure in ["recall", "precision"]:
                diffs.append({"model": tag, "prompt": prompt, "measure": measure,
                              **paired_diffs(stretch[stretch["prompt"] == prompt], measure)})
    table = pd.concat(main_tables, ignore_index=True)
    table.to_csv(config.SIZE_METRICS_PATH, index=False)
    pd.concat(stretch_tables, ignore_index=True).to_csv(config.SIZE_STRETCH_METRICS_PATH, index=False)
    diffs = pd.DataFrame(diffs).round(3)
    diffs.to_csv(config.SIZE_STRETCH_DIFFS_PATH, index=False)
    mixing = pd.DataFrame(mixing).round(3)
    mixing.to_csv(config.SIZE_MIXING_DIFFS_PATH, index=False)
    plot_sizes(table)
    print(table[["model", "level", "prompt", "recall", "precision", "f1", "unsupported_per_summary",
                 "summary_words", "compression", "time_s", "hit_max_tokens"]].to_string(index=False))
    print(diffs.to_string(index=False))
    print(mixing.to_string(index=False))
    print(f"Wrote {config.SIZE_METRICS_PATH}, {config.SIZE_STRETCH_METRICS_PATH}, "
          f"{config.SIZE_STRETCH_DIFFS_PATH}, {config.SIZE_MIXING_DIFFS_PATH}, {config.SIZE_CHART_PATH}")


def main():
    """Score all summaries (default), or make / summarise the Day 4 manual-check sheets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--make-sheets", action="store_true", help="write the blank tagging and checklist sheets")
    parser.add_argument("--summarise-tags", action="store_true", help="count the filled-in error tags")
    parser.add_argument("--stretch", action="store_true", help="score targeted vs random mixing (Day 5 stretch)")
    parser.add_argument("--model", choices=list(config.MODELS), default="1b",
                        help="which model's summaries to score (3b/8b: the model-size extension)")
    parser.add_argument("--compare-sizes", action="store_true", help="combine all models' scores (run after scoring each)")
    args = parser.parse_args()
    paths = config.model_paths(args.model)
    if args.compare_sizes:
        return compare_sizes()
    if args.stretch:
        return score_stretch(paths)
    if args.make_sheets:
        return make_sheets()
    if args.summarise_tags:
        return summarise_tags()

    scores = score_all(paths["generations"])
    with open(paths["scores"], "w", encoding="utf-8") as f:
        for row in scores:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    table = metrics_table(scores)
    table.to_csv(paths["metrics"], index=False)
    if args.model == "1b":
        plot(table)  # the main chart; larger models are charted by --compare-sizes
    print(table.to_string(index=False))
    print(f"Wrote {paths['scores']}, {paths['metrics']}")


if __name__ == "__main__":
    main()
