"""Content preservation: how close is each summary to the human-written XL-Sum reference? (ROUGE, BERTScore)

Fact recall/precision (score.py) and NLI (verify.py) compare a summary with the article.
This script compares it with the BBC's own one-line summary that ships with XL-Sum, the
standard content measure for summarisation papers.

  ROUGE-1 / ROUGE-2 / ROUGE-L F1: word overlap. Hindi has no stemmer, so text is only
      normalised the same way as fact matching (score.match_key: Unicode, digits, nukta,
      punctuation) and split on spaces. The default rouge_score tokenizer drops every
      non-Latin character, so it must not be used for Hindi.
  BERTScore F1: similarity of contextual embeddings (CONTENT_BERT_MODEL), raw, not rescaled.

Every summary is the same first-3-sentence cut as score.py. Two article baselines are
scored too: Lead-1 and Lead-3 (the article's first 1 or 3 sentences).

Usage:  .venv/bin/python content.py              # -> results/content_scores.jsonl, content_metrics.csv, content_diffs.csv
        .venv/bin/python content.py --summarise  # rebuild the two tables from content_scores.jsonl only
"""

import argparse
import json

import pandas as pd
import torch
from rouge_score import rouge_scorer

import config
from score import match_key, paired_diffs
from verify import sentences


class Tokenizer:
    """Whitespace tokenizer over the fact-matching normalisation (works for Devanagari and Latin)."""

    def tokenize(self, text):
        return match_key(text).split()


def lead(text, n):
    return " ".join(sentences(text)[:n])


def collect():
    """Every summary to score: all models' main and stretch runs, plus the Lead baselines."""
    rows = []
    for tag in config.MODELS:
        paths = config.model_paths(tag)
        for kind, key in [("main", "scores"), ("stretch", "stretch_scores")]:
            for s in map(json.loads, open(paths[key], encoding="utf-8")):
                rows.append({"model": tag, "run": kind, "id": s["id"],
                             "level": str(s.get("level", s.get("condition"))),
                             "prompt": s["prompt"], "summary": s["scored_summary"]})
    for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8")):
        for n in (1, 3):
            rows.append({"model": f"lead-{n}", "run": "baseline", "id": a["id"], "level": "0",
                         "prompt": "none", "summary": lead(a["text"], n)})
    return rows


def score(rows):
    from bert_score import score as bertscore
    refs = {a["id"]: a["reference_summary"] for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=Tokenizer())
    for r in rows:
        s = scorer.score(refs[r["id"]], r["summary"])
        r.update({k: s[k].fmeasure for k in ["rouge1", "rouge2", "rougeL"]})
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    _, _, f1 = bertscore([r["summary"] for r in rows], [refs[r["id"]] for r in rows],
                         model_type=config.CONTENT_BERT_MODEL, batch_size=config.CONTENT_BATCH,
                         device=device, verbose=True)
    for r, f in zip(rows, f1.tolist()):
        r["bertscore"] = f
    return rows


MEASURES = ["rouge1", "rouge2", "rougeL", "bertscore"]


def summarise():
    df = pd.DataFrame(map(json.loads, open(config.CONTENT_SCORES_PATH, encoding="utf-8")))
    order = list(config.MODELS) + ["lead-1", "lead-3"]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    table = df.groupby(["model", "run", "level", "prompt"], observed=True).agg(
        n=("id", "size"), **{m: (m, "mean") for m in MEASURES}).reset_index().round(4)
    table.to_csv(config.CONTENT_METRICS_PATH, index=False)

    diffs = []
    for tag in config.MODELS:
        main = df[(df["model"] == tag) & (df["run"] == "main")].assign(level=lambda d: d["level"].astype(int))
        stretch = df[(df["model"] == tag) & (df["run"] == "stretch")]
        for m in MEASURES:
            for prompt in config.PROMPTS:
                diffs.append({"model": tag, "comparison": f"L80-L0 {prompt}", "measure": m,
                              **paired_diffs(main[main["prompt"] == prompt], m, "level", 80, 0)})
                diffs.append({"model": tag, "comparison": f"targeted-random {prompt}", "measure": m,
                              **paired_diffs(stretch[stretch["prompt"] == prompt], m, "level")})
            for level in config.SCALE_LEVELS:
                diffs.append({"model": tag, "comparison": f"checklist-plain L{level}", "measure": m,
                              **paired_diffs(main[main["level"] == level], m, "prompt", "checklist", "plain")})
    diffs = pd.DataFrame(diffs).round(4)
    diffs.to_csv(config.CONTENT_DIFFS_PATH, index=False)
    print(table.to_string(index=False))
    print(diffs.to_string(index=False))
    print(f"Wrote {config.CONTENT_METRICS_PATH}, {config.CONTENT_DIFFS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true", help="only rebuild the tables from content_scores.jsonl")
    if not parser.parse_args().summarise:
        rows = score(collect())
        with open(config.CONTENT_SCORES_PATH, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarise()


if __name__ == "__main__":
    main()
