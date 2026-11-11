"""Automatic meaning check: is each summary sentence entailed by the original article? (NLI)

Precision and recall (score.py) only check single facts (names, numbers, dates). They cannot
see a summary that uses the right names in the wrong roles. This script asks a multilingual
NLI model, sentence by sentence, whether the article supports the summary.

For each summary sentence (the same first-3-sentence cut as score.py):
  1. split the original article into overlapping windows of NLI_WINDOW sentences,
  2. keep the NLI_TOP_WINDOWS windows with the most word overlap with the sentence,
  3. run NLI (premise = window, hypothesis = sentence) on each;
  supported    = the best entailment probability >= NLI_THRESHOLD,
  contradicted = not supported, and the most-overlapping window contradicts it (>= NLI_THRESHOLD).

Usage:  .venv/bin/python verify.py --validate     # check the method against the 48 hand-tagged summaries
        .venv/bin/python verify.py                # all models -> results/nli_scores.jsonl, nli_metrics.csv, nli_diffs.csv
        .venv/bin/python verify.py --summarise    # rebuild the two tables from nli_scores.jsonl only
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from extract import SENTENCE_RE


def sentences(text):
    """Split text into sentences with the same rule the scorer uses."""
    return [m.group().strip() for line in text.split("\n") for m in SENTENCE_RE.finditer(line) if m.group().strip()]


def windows(article_text):
    """Overlapping windows of NLI_WINDOW consecutive article sentences."""
    sents = sentences(article_text)
    step = max(config.NLI_WINDOW - 1, 1)
    return [" ".join(sents[i:i + config.NLI_WINDOW]) for i in range(0, max(len(sents) - 1, 1), step)]


def best_windows(hypothesis, wins):
    """The NLI_TOP_WINDOWS windows sharing the most words with the hypothesis, best first."""
    words = set(hypothesis.split())
    ranked = sorted(wins, key=lambda w: len(words & set(w.split())), reverse=True)
    return ranked[: config.NLI_TOP_WINDOWS]


class NLI:
    """A multilingual NLI classifier returning (entailment, contradiction) probabilities."""

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]

    def __call__(self, pairs):
        out = []
        for i in range(0, len(pairs), config.NLI_BATCH):
            batch = pairs[i:i + config.NLI_BATCH]
            enc = self.tokenizer([p for p, _ in batch], [h for _, h in batch], truncation="only_first",
                                 max_length=config.NLI_MAX_TOKENS, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                probs = self.model(**enc).logits.float().softmax(-1).cpu().numpy()
            out += [(p[self.entail], p[self.contra]) for p in probs]
        return out


def check_summaries(items, articles, nli):
    """items: dicts with id + scored_summary. Adds nli_support, nli_contradicted, nli_sentences."""
    pairs, index = [], []   # index: (item number, sentence number, window rank)
    for n, item in enumerate(items):
        wins = windows(articles[item["id"]]["text"])
        for s, sent in enumerate(sentences(item["scored_summary"])):
            for rank, win in enumerate(best_windows(sent, wins)):
                pairs.append((win, sent))
                index.append((n, s, rank))
    probs = nli(pairs)
    per_sentence = {}
    for (n, s, rank), (ent, con) in zip(index, probs):
        best = per_sentence.setdefault((n, s), {"entail": 0.0, "contra_top": 0.0})
        best["entail"] = max(best["entail"], float(ent))
        if rank == 0:
            best["contra_top"] = float(con)
    for n, item in enumerate(items):
        sents = [v for (m, _), v in sorted(per_sentence.items()) if m == n]
        supported = [v["entail"] >= config.NLI_THRESHOLD for v in sents]
        contradicted = [not sup and v["contra_top"] >= config.NLI_THRESHOLD for sup, v in zip(supported, sents)]
        item["nli_sentences"] = len(sents)
        item["nli_support"] = float(np.mean(supported)) if sents else None
        item["nli_contradicted"] = int(any(contradicted))
        item["nli_min_entail"] = min((v["entail"] for v in sents), default=None)
    return items


def auc(scores, labels):
    """Probability that a random error summary scores lower than a random clean one (ties count half)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    wins = sum((p < q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def validate(nli, articles):
    """Compare the NLI scores with the final hand tags on the 48 tagged summaries (Llama 1B)."""
    tags = pd.read_csv(config.ERROR_TAGS_PATH, encoding="utf-8-sig")
    key = pd.read_csv(config.ERROR_TAGS_KEY_PATH)
    tags = tags.merge(key, on=["item", "article_id"])
    scores = {(s["id"], s["level"], s["prompt"]): s for s in map(json.loads, open(config.SCORES_PATH, encoding="utf-8"))}
    items = [dict(scores[(r.article_id, r.level, r.prompt)], tags=str(r.tags)) for r in tags.itertuples()]
    check_summaries(items, articles, nli)
    df = pd.DataFrame(items)
    factual = ["entity swap", "number drift", "date shift", "unsupported addition"]
    df["factual_error"] = df["tags"].apply(lambda t: any(c in t for c in factual))
    df["entity_swap"] = df["tags"].str.contains("entity swap")
    rows = []
    for label in ["factual_error", "entity_swap"]:
        rows.append({
            "hand_label": label, "n_with": int(df[label].sum()), "n_without": int((~df[label]).sum()),
            "support_with": df.loc[df[label], "nli_support"].mean(), "support_without": df.loc[~df[label], "nli_support"].mean(),
            "auc_support": auc(df["nli_support"], df[label]), "auc_min_entail": auc(df["nli_min_entail"], df[label]),
            "contradicted_with": df.loc[df[label], "nli_contradicted"].mean(),
            "contradicted_without": df.loc[~df[label], "nli_contradicted"].mean(),
        })
    table = pd.DataFrame(rows).round(3)
    table.to_csv(config.NLI_VALIDATION_PATH, index=False)
    print(table.to_string(index=False))
    print(f"Wrote {config.NLI_VALIDATION_PATH}")


def run_all(nli, articles):
    """NLI scores for every model's main summaries -> nli_scores.jsonl and the model x level x prompt table."""
    rows = []
    for tag in config.MODELS:
        scores = [s for s in map(json.loads, open(config.model_paths(tag)["scores"], encoding="utf-8"))
                  if s["level"] in config.SCALE_LEVELS]
        items = [{"model": tag, "id": s["id"], "level": s["level"], "prompt": s["prompt"],
                  "scored_summary": s["scored_summary"]} for s in scores]
        rows += check_summaries(items, articles, nli)
        print(f"{tag}: {len(items)} summaries checked")
    with open(config.NLI_SCORES_PATH, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summarise()


def summarise():
    """From nli_scores.jsonl: the model x level x prompt table and paired intervals (no NLI rerun needed)."""
    from score import paired_diffs  # same seeded bootstrap as the other metrics
    df = pd.DataFrame(map(json.loads, open(config.NLI_SCORES_PATH, encoding="utf-8")))
    diffs = []
    for tag in config.MODELS:
        part = df[df["model"] == tag]
        for measure in ["nli_support", "nli_contradicted"]:
            for prompt in config.PROMPTS:
                diffs.append({"model": tag, "comparison": f"L80-L0 {prompt}", "measure": measure,
                              **paired_diffs(part[part["prompt"] == prompt], measure, "level", 80, 0)})
            for level in config.SCALE_LEVELS:
                diffs.append({"model": tag, "comparison": f"checklist-plain L{level}", "measure": measure,
                              **paired_diffs(part[part["level"] == level], measure, "prompt", "checklist", "plain")})
    diffs = pd.DataFrame(diffs).round(3)
    diffs.to_csv(config.NLI_DIFFS_PATH, index=False)
    df["prompt"] = pd.Categorical(df["prompt"], categories=config.PROMPTS, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=list(config.MODELS), ordered=True)
    table = df.groupby(["model", "level", "prompt"], observed=True).agg(
        n=("id", "size"), nli_support=("nli_support", "mean"), nli_contradicted=("nli_contradicted", "mean"),
    ).reset_index().round(3)
    table.to_csv(config.NLI_METRICS_PATH, index=False)
    print(table.to_string(index=False))
    print(diffs.to_string(index=False))
    print(f"Wrote {config.NLI_METRICS_PATH}, {config.NLI_DIFFS_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="compare with the 48 hand-tagged summaries")
    parser.add_argument("--summarise", action="store_true", help="only rebuild the tables from nli_scores.jsonl")
    args = parser.parse_args()
    if args.summarise:
        return summarise()
    articles = {a["id"]: a for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    nli = NLI()
    print(f"NLI model {config.NLI_MODEL} on {nli.device}")
    validate(nli, articles) if args.validate else run_all(nli, articles)


if __name__ == "__main__":
    main()
