"""Prompting-strategy extension: compare four prompts on the same articles, models and levels.

  plain, checklist   existing summaries (results/scores.jsonl, results/scale/gemma9b/scores.jsonl)
  instructed         checklist + explicit factuality rules              (generate.py --prompting)
  verified           instructed draft, checked and revised by the model (generate.py --prompting)

For every model in PROMPTING_MODELS and level in SCALE_LEVELS this script scores the new
summaries with the same three checks as the rest of the project (fact recall/precision/F1,
sentence-level NLI, ROUGE/BERTScore against the XL-Sum reference), reuses the stored scores
of the plain and checklist summaries, and writes paired comparisons over the 100 articles.

Usage:  .venv/bin/python prompting.py              # score new summaries (NLI + BERTScore: ~10 min on Apple GPU)
        .venv/bin/python prompting.py --summarise  # rebuild tables and chart from the stored scores only
Outputs in results/prompting/: <model>/scores.jsonl, prompting_metrics.csv, prompting_diffs.csv,
prompting_chart.png.
"""

import argparse
import json
import re
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config
from score import add_f1, metrics_table, paired_diffs, score_all

MEASURES = ["recall", "precision", "f1", "n_unsupported", "nli_support", "nli_contradicted", "rouge1", "bertscore"]
# Comparisons a - b on the same article and level.
PROMPT_PAIRS = [("instructed", "checklist"), ("verified", "instructed"), ("verified", "checklist"),
                ("instructed", "plain"), ("verified", "plain")]


def paths(tag):
    d = config.PROMPTING_DIR / tag
    return {"generations": d / "generations.jsonl", "scores": d / "scores.jsonl"}


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def score_new():
    """Fact, NLI and content scores for every instructed / verified summary -> <model>/scores.jsonl."""
    from content import score as content_score
    from verify import NLI, check_summaries
    articles = {a["id"]: a for a in read_jsonl(config.CHECKLISTS_PATH)}
    nli = NLI()
    for tag in config.PROMPTING_MODELS:
        gens = read_jsonl(paths(tag)["generations"])
        rows = score_all(paths(tag)["generations"])
        # Did the verification pass change the draft? (compared after the same 3-sentence cut)
        for row, g in zip(rows, gens):
            row["changed_draft"] = None if g["prompt"] != "verified" else int(row["scored_summary"] != g["draft"])
            row["check_time_s"] = g.get("check_time_s")
        check_summaries(rows, articles, nli)
        content = content_score([{"id": r["id"], "summary": r["scored_summary"]} for r in rows])
        for row, c in zip(rows, content):
            row.update({m: c[m] for m in ["rouge1", "rouge2", "rougeL", "bertscore"]})
        with open(paths(tag)["scores"], "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{tag}: {len(rows)} summaries scored")


def all_scores():
    """One frame: every model x level x prompt (four prompts), with fact, NLI and content measures."""
    nli = pd.DataFrame(read_jsonl(config.NLI_SCORES_PATH))
    content = pd.DataFrame(read_jsonl(config.CONTENT_SCORES_PATH))
    content = content[content["run"] == "main"].assign(level=lambda d: d["level"].astype(int))
    frames = []
    for tag in config.PROMPTING_MODELS:
        old = pd.DataFrame(read_jsonl(config.model_paths(tag)["scores"]))
        old = old[old["level"].isin(config.SCALE_LEVELS)]
        keys = ["id", "level", "prompt"]
        old = old.merge(nli[nli["model"] == tag][keys + ["nli_support", "nli_contradicted"]], on=keys, how="left")
        old = old.merge(content[content["model"] == tag][keys + ["rouge1", "rouge2", "rougeL", "bertscore"]],
                        on=keys, how="left")
        new = pd.DataFrame(read_jsonl(paths(tag)["scores"]))
        frames.append(pd.concat([old, new], ignore_index=True).assign(model=tag))
    return add_f1(pd.concat(frames, ignore_index=True))


def summarise():
    df = all_scores()
    tables, diffs = [], []
    for tag in config.PROMPTING_MODELS:
        part = df[df["model"] == tag]
        table = metrics_table(part.drop(columns=["f1"]).to_dict("records"))
        extra = part.groupby(["level", "prompt"]).agg(
            nli_support=("nli_support", "mean"), nli_contradicted=("nli_contradicted", "mean"),
            rouge1=("rouge1", "mean"), bertscore=("bertscore", "mean"),
            changed_draft=("changed_draft", "mean")).reset_index()
        table["prompt"] = table["prompt"].astype(str)
        tables.append(table.merge(extra, on=["level", "prompt"]).assign(model=tag))
        for level in config.SCALE_LEVELS:
            at = part[part["level"] == level]
            for a, b in PROMPT_PAIRS:
                for m in MEASURES:
                    diffs.append({"model": tag, "level": level, "comparison": f"{a}-{b}", "measure": m,
                                  **paired_diffs(at.dropna(subset=[m]), m, "prompt", a, b)})
        for prompt in config.ALL_PROMPTS:
            for m in ["n_unsupported", "precision", "nli_contradicted"]:
                diffs.append({"model": tag, "level": "80-0", "comparison": f"L80-L0 {prompt}", "measure": m,
                              **paired_diffs(part[part["prompt"] == prompt].dropna(subset=[m]), m, "level", 80, 0)})
    table = pd.concat(tables, ignore_index=True)
    order = {p: i for i, p in enumerate(config.ALL_PROMPTS)}
    table = table.sort_values(["model", "level", "prompt"], key=lambda s: s.map(order) if s.name == "prompt" else s,
                              kind="stable").reset_index(drop=True)
    cols = ["model", "level", "prompt", "n", "recall", "recall_numbers", "precision", "f1", "unsupported_per_summary",
            "nli_support", "nli_contradicted", "rouge1", "bertscore", "summary_words", "compression",
            "hit_max_tokens", "time_s", "changed_draft"]
    table = table[cols].round(3)
    table.to_csv(config.PROMPTING_METRICS_PATH, index=False)
    diffs = pd.DataFrame(diffs).round(3)
    diffs.to_csv(config.PROMPTING_DIFFS_PATH, index=False)
    plot(table)
    print(table.to_string(index=False))
    print(diffs[diffs["comparison"].isin(["instructed-checklist", "verified-checklist", "verified-instructed"])
                | diffs["comparison"].str.startswith("L80")].to_string(index=False))
    print(f"Wrote {config.PROMPTING_METRICS_PATH}, {config.PROMPTING_DIFFS_PATH}, {config.PROMPTING_CHART_PATH}")


def plot(table):
    """Rows = models, columns = measures, x = mixing level, one line per prompt."""
    measures = [("recall", "Fact recall"), ("unsupported_per_summary", "Unsupported facts / summary"),
                ("nli_contradicted", "Summaries with a contradicted sentence")]
    models = config.PROMPTING_MODELS
    fig, axes = plt.subplots(len(models), len(measures), figsize=(12, 3.4 * len(models)), sharex=True,
                             sharey="col", facecolor="white")
    for r, tag in enumerate(models):
        for c, (m, title) in enumerate(measures):
            ax = axes[r, c]
            for prompt in config.ALL_PROMPTS:
                part = table[(table["model"] == tag) & (table["prompt"] == prompt)]
                ax.plot(part["level"], part[m], marker="o", linewidth=2, label=prompt,
                        color=config.PROMPT_COLORS_ALL[prompt])
            ax.set_xticks(config.SCALE_LEVELS)
            ax.grid(axis="y", color="#e5e5e5")
            ax.spines[["top", "right"]].set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=11, loc="left")
            if r == len(models) - 1:
                ax.set_xlabel("mixing level (%)")
            if c == 0:
                ax.set_ylabel(config.MODELS[tag].split("/")[-1].replace("-Instruct", "").replace("-it", ""))
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle("Four prompting strategies vs mixing level (100 articles per point)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(config.PROMPTING_CHART_PATH, dpi=200, facecolor="white")


# A list marker at the start of a line: "1. ", "2) ", "- ", "* ", "• ".
LIST_MARKER_RE = re.compile(r"^[ \t]*(?:\d{1,2}[.)]|[-*•])[ \t]+", re.M)


def list_marker_check():
    """Sensitivity check: Llama 1B often answers the verify pass as a numbered list. The scorer then
    counts "1." as a sentence and "1", "2" as unsupported numbers. Rescore the new prompts with list
    markers removed (the project's scoring rule itself is unchanged) -> list_marker_check.csv."""
    out = []
    for tag in config.PROMPTING_MODELS:
        gens = read_jsonl(paths(tag)["generations"])
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as tmp:
            for g in gens:
                tmp.write(json.dumps({**g, "summary": LIST_MARKER_RE.sub("", g["summary"])}, ensure_ascii=False) + "\n")
        stripped = add_f1(pd.DataFrame(score_all(tmp.name)))
        stripped["had_markers"] = [bool(LIST_MARKER_RE.search(g["summary"])) for g in gens]
        table = stripped.groupby(["level", "prompt"]).agg(
            n_with_markers=("had_markers", "sum"), recall=("recall", "mean"), precision=("precision", "mean"),
            f1=("f1", "mean"), unsupported_per_summary=("n_unsupported", "mean"),
            summary_words=("summary_words", "mean")).reset_index()
        out.append(table.assign(model=tag))
    table = pd.concat(out, ignore_index=True).round(3)
    table.to_csv(config.PROMPTING_DIR / "list_marker_check.csv", index=False)
    print(table.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarise", action="store_true", help="only rebuild the tables and chart")
    parser.add_argument("--list-check", action="store_true", help="only the list-marker sensitivity check")
    args = parser.parse_args()
    if args.list_check:
        return list_marker_check()
    if not args.summarise:
        score_new()
    summarise()
    list_marker_check()


if __name__ == "__main__":
    main()
