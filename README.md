# FactIndic

As Hindi news text becomes more code-mixed with English, do LLM summaries lose facts or add wrong ones? And does putting an automatically extracted fact checklist in the prompt help? This is a 5-day, inference-only student experiment; the full plan is in [PLAN.md](PLAN.md), and every choice it left open is in [DECISIONS.md](DECISIONS.md).

**Status:** Plan complete (Days 1-5). `REPORT.md` is final; slides: https://claude.ai/artifact/PhyZ5TeEszC3owEEK2NuNx (private until shared). Stretch (targeted vs random mixing) is done: 400 summaries generated on Kaggle, scored, and written up in REPORT.md section 6 and on the deck. Model-size extension (Llama-3.2-3B and Llama-3.1-8B; levels 0/40/80 + stretch, 1,000 summaries each) is done: generated on Kaggle notebooks `factindic-3b` / `factindic-8b`, scored, and written up in REPORT.md section 7 and on the deck. Second model family, Gemma 2 (2B, 9B; same scope), is done: Kaggle notebooks `factindic-gemma2b` / `factindic-gemma9b`, REPORT.md section 8 and a deck slide. Paired bootstrap intervals (levels 0 vs 80, checklist vs plain) now back the main claims. Checked against the mentor's brief (CSET 346 PJ_06): F1, compression and inference time are reported, and `verify.py` adds an automatic sentence-level meaning check (NLI). The IEEE-style paper is still to do. Source code, data and results are public on GitHub: https://github.com/Swapnil-Pramanik/FactIndic

## How to run (in order)

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
# IndicNER and Llama-3.2 are gated: accept both licences on huggingface.co, then
.venv/bin/hf auth login

.venv/bin/python extract.py        # -> data/article_ids.txt, data/checklists.jsonl
.venv/bin/python mix.py            # -> data/mixed.jsonl, data/translations.json, data/mixed_samples_for_review.txt
.venv/bin/python tests/checks.py   # all checks must pass
# Day 2, on a T4 GPU (see below):
python generate.py --dry-run       # 10 articles -> results/dry_run.jsonl (100 rows)
python generate.py                 # all 100    -> results/generations.jsonl (1,000 rows); rerun to resume
.venv/bin/python score.py           # -> results/scores.jsonl, metrics.csv, chart.png (~40 s, CPU)
# Day 4:
.venv/bin/python score.py --make-sheets     # blank results/error_tags.csv (+ key) and checklist_check.csv; never overwrites
#   fill in the tags column (several tags separated by ';', 'none' if no error) and the checklist counts
.venv/bin/python score.py --summarise-tags  # -> results/error_tag_counts.csv
# Stretch (targeted vs random mixing):
.venv/bin/python mix.py --stretch           # -> data/stretch_mixed.jsonl (200 rows)
python generate.py --stretch                # on the T4 -> results/stretch_generations.jsonl (400 rows), resumable
.venv/bin/python score.py --stretch         # -> results/stretch_scores.jsonl, stretch_metrics.csv
# Model-size extension (3b = Llama-3.2-3B, 8b = Llama-3.1-8B; levels 0/40/80 only):
python generate.py --model 3b               # on the T4 -> results/scale/3b/generations.jsonl (600 rows)
python generate.py --model 3b --stretch     # -> results/scale/3b/stretch_generations.jsonl (400 rows)
#   same for --model 8b / gemma2b / gemma9b; Kaggle notebooks factindic-gemma2b / factindic-gemma9b too; factindic-3b / factindic-8b run both commands with an 11h15m deadline
.venv/bin/python score.py --model 3b        # -> results/scale/3b/scores.jsonl, metrics.csv (same for 8b)
.venv/bin/python score.py --model 3b --stretch   # -> results/scale/3b/stretch_scores.jsonl, stretch_metrics.csv
.venv/bin/python score.py --compare-sizes   # all five models -> results/scale/size_*.csv, size_chart.png
# Automatic meaning check (NLI, CPU/Apple GPU, ~1 min for the validation, longer for all models):
.venv/bin/python verify.py --validate       # NLI vs the 48 hand tags -> results/nli_validation.csv
.venv/bin/python verify.py                  # all models, levels 0/40/80 -> results/nli_scores.jsonl, nli_metrics.csv, nli_diffs.csv (~20 min on Apple GPU)
```

Each script reads only the files written by the step before it, so any step can be rerun on its own.

### Running generation on Kaggle (or Colab)

`generate.py` needs only `config.py`, `prompts.py`, `generate.py`, `data/checklists.jsonl` and `data/mixed.jsonl`.

**On Colab:**

1. Upload `FactIndic_colab.zip` (those five files plus `requirements.txt`) to Google Drive, and unzip it to `MyDrive/FactIndic`. Running from Drive means `results/` survives a disconnect.
2. In Colab: *Runtime -> Change runtime type -> T4 GPU*. Add your Hugging Face token under *Secrets* (key icon) as `HF_TOKEN` and enable notebook access. Never paste the token into a cell.
3. Run these cells:

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
%cd /content/drive/MyDrive/FactIndic
!pip install -q transformers==5.22.2 bitsandbytes accelerate
!pip list 2>/dev/null | grep -iE "^(torch|transformers|bitsandbytes|accelerate) "   # note these versions
!python generate.py --dry-run
```

4. After the dry run has been reviewed: `!python generate.py`. If the session dies, reconnect, rerun the setup cell, then run the same command again; it skips finished rows.

**On Kaggle (the platform this project uses):**

1. *Create -> New Dataset*, upload `FactIndic_colab.zip` (Kaggle unzips it), name it `factindic`, keep it private.
2. New notebook. *Settings*: Accelerator **GPU T4 x2** (only one GPU is used), Internet **On**. *Add Input*: the `factindic` dataset. *Add-ons -> Secrets*: `HF_TOKEN`, attached to the notebook.
3. Cells:

```python
from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
# /kaggle/input is read-only, so work in /kaggle/working
!mkdir -p /kaggle/working/FactIndic && cp -r /kaggle/input/factindic/* /kaggle/working/FactIndic/
%cd /kaggle/working/FactIndic
!pip install -q transformers==5.22.2 bitsandbytes accelerate
!pip list 2>/dev/null | grep -iE "^(torch|transformers|bitsandbytes|accelerate) "
!python generate.py --dry-run      # then download results/dry_run.jsonl from the Output panel
```

4. Full run: change the last line to `!python generate.py` and use *Save Version -> Save & Run All (Commit)*. It runs in the background (12 h limit) and keeps `results/` as the version's output. Don't rely on an interactive session for it: `/kaggle/working` is wiped when that session ends, so resume cannot help there.

Without a CUDA GPU, `generate.py --dry-run` falls back to full precision on CPU/Apple GPU (field `setup` = `mps-fp32` / `cpu-fp32`). That is for testing code only, and the full run refuses to start without CUDA.

## Features

| Feature | Where | Status |
|---|---|---|
| Seeded selection of 100 Hindi XL-Sum test articles (150-600 words) | `extract.py` | done |
| Fact checklist: entities (IndicNER), numbers and dates (regex, Devanagari digits too) | `extract.py` | done |
| Code-mixing of nouns and adjectives at 0/20/40/60/80 %, nested, facts protected | `mix.py` | done |
| Sanity checks | `tests/checks.py` | done (extractor, mixer, scorer) |
| Summaries with `plain` / `checklist` prompts, resumable | `generate.py`, `prompts.py` | done (1,000 rows) |
| Recall / precision / length table and chart | `score.py` | done |
| Blind error-tagging sheet + tag counts, checklist quality sheet | `score.py --make-sheets / --summarise-tags` | done (student-tagged, Claude-audited) |
| Stretch: targeted (fact words) vs random mixing at equal word counts | `mix.py --stretch`, `generate.py --stretch`, `score.py --stretch` | done (400 rows; REPORT.md section 6) |
| Model-size extension: same pipeline on Llama-3.2-3B and Llama-3.1-8B | `generate.py --model 3b/8b`, `score.py --model 3b/8b [--stretch]`, `score.py --compare-sizes` | Llama 3B/8B done (2 x 1,000 rows; REPORT.md section 7); Gemma 2 2B/9B (`--model gemma2b/gemma9b`) done (2 x 1,000 rows; REPORT.md section 8) |
| F1, compression ratio, inference time (brief's metric list) | `score.py` (`add_f1`, `metrics_table`) | done |
| Automatic meaning check: sentence-level NLI, validated against the hand tags | `verify.py` | done |
| Report | `REPORT.md` | final (IEEE paper version still to do) |

## Architecture

```
XL-Sum (HF Parquet) ──> extract.py ──> data/checklists.jsonl ──> mix.py ──> data/mixed.jsonl
                          │ IndicNER                               │ Stanza POS + MUSE dict + wordfreq
                          │ regex numbers/dates                    │
                                                                   v       prompts.py (plain / checklist)
                                                                                   │
                  results/metrics.csv, chart.png <── score.py <── results/generations.jsonl <── generate.py (Llama-3.2-1B, T4)
```

Stages talk only through JSONL/CSV files on disk. All settings live in `config.py`.

## Project structure

```
PLAN.md                 # the fixed plan (do not change)
DECISIONS.md            # every open choice, one line each
README.md               # this file
REPORT.md               # the write-up (all result numbers from results/metrics.csv)
FUTURE_WORK.md          # ideas outside the plan's scope, not built
requirements.txt        # pinned library versions
FactIndic_colab.zip     # the 6 files generate.py needs (incl. data/stretch_mixed.jsonl) + requirements.txt, for Kaggle/Colab
                        #   (rebuild it if config.py, prompts.py, generate.py or data/ change)
config.py               # all settings and paths
extract.py              # load + filter data, build checklists
mix.py                  # code-mixing at each level
prompts.py              # the plain and checklist prompt text
generate.py             # Llama summaries (1B main; --model 3b/8b for the size extension), resumable (run on a T4)
score.py                # cut to 3 sentences, recall/precision/F1, tables + charts, model comparison
verify.py               # sentence-level NLI meaning check (mDeBERTa XNLI), validated on the hand tags
.gitignore              # ignores .venv, caches, data/dict (downloaded) and FactIndic_colab.zip (rebuilt)
tests/checks.py         # plain-assert sanity checks
data/
  dict/hi-en.txt        # MUSE Hindi->English dictionary (downloaded by mix.py)
  article_ids.txt       # the 100 chosen article ids
  checklists.jsonl      # one article per line: text + checklist + protected spans
  mixed.jsonl           # one (article, level) per line: 500 rows
  translations.json     # every Hindi->English replacement actually used
  mixed_samples_for_review.txt  # 4 articles x 5 levels, for the Day 1 eyeball check
  stretch_mixed.jsonl   # stretch: targeted + random version of each article (200 rows)
results/                # created by generate.py
  dry_run.jsonl         # 10 articles x 5 levels x 2 prompts = 100 rows
  generations.jsonl     # the real run: 1,000 rows
  scores.jsonl          # one row per summary: scored text, recall, precision, unsupported facts
  metrics.csv           # level x prompt table (every number in the report comes from here)
  chart.png             # recall and precision vs mixing level, one line per prompt
  error_tags.csv        # FINAL error tags (48 summaries): student tags audited by Claude, student_tags + audit_note columns
  error_tags_filled.csv # the student's original tags, untouched
  error_tags_key.csv    # item -> level, prompt for that sheet
  error_tag_counts.csv  # tag counts per (level, prompt), from --summarise-tags
  nli_validation.csv    # verify.py --validate: NLI support vs the 48 hand tags (AUC etc.)
  nli_scores.jsonl / nli_metrics.csv  # verify.py: NLI support per summary; model x level x prompt table
  nli_diffs.csv         # NLI: level 80 - level 0 and checklist - plain per model, paired, 95% intervals
  checklist_check.csv   # FINAL checklist-quality counts (20 articles), audited, with missed_items + audit_note
  checklist_check_filled.csv  # the student's original counts, untouched
  stretch_generations.jsonl   # 400 stretch summaries (Kaggle T4, cuda-4bit)
  stretch_scores.jsonl / stretch_metrics.csv  # stretch scores per summary; condition x prompt table
  scale/
    <3b|8b|gemma2b|gemma9b>/  # per model: generations.jsonl (600, levels 0/40/80), stretch_generations.jsonl (400),
                        #   scores.jsonl, metrics.csv, stretch_scores.jsonl, stretch_metrics.csv
    size_metrics.csv    # level x prompt x model, levels 0/40/80 (1B from the main scores)
    size_stretch_metrics.csv  # condition x prompt x model
    size_stretch_diffs.csv    # targeted - random per model/prompt/measure, paired, 95% bootstrap interval
    size_mixing_diffs.csv     # level 80 - level 0 and checklist - plain per model, paired, 95% bootstrap interval
    size_chart.png      # recall and precision vs mixing level, one column per model (5 models)
.venv/                  # local virtual environment (not part of the project)
```

Nothing planned. Ideas for later are in `FUTURE_WORK.md`.

## Layer breakdown

**`extract.py`**

| Function | Responsibility |
|---|---|
| `normalize_text` | Unicode-normalise Devanagari (nukta forms) and collapse whitespace |
| `normalize_digits` / `normalize_number` | Devanagari -> Western digits; drop thousands commas |
| `find_dates` / `find_numbers` | regex spans; numbers inside dates are skipped |
| `load_ner` | IndicNER pipeline with no aggregation (its built-in grouping breaks on this model) |
| `word_labels` / `merge_words` / `trim` | label each whitespace word by its first sub-word, join B-/I- runs, strip edge punctuation |
| `find_entities` | run NER per sentence; keep PER/LOC/ORG with score >= 0.5 that are not just function words |
| `build_checklist` | the checklist dict plus the spans the mixer must protect |
| `unique` | drop duplicates, keep first-seen order |
| `load_articles` | load Parquet, length filter, seeded shuffle, take 100 |

**`mix.py`**

| Function | Responsibility |
|---|---|
| `load_dictionary` | MUSE hi-en -> one English word per Hindi word (most frequent, Zipf >= 3.0) |
| `strip_nukta` / `translate` | look up lemma first for singular words, exact form first for plurals, then both without nukta |
| `load_tagger` / `tag_words` | Stanza Hindi tokenize + POS + lemma + number (sing./plural), with character offsets |
| `is_eligible` / `eligible_words` | nouns and adjectives that are not facts and can be translated (verbs are never replaced, see DECISIONS.md) |
| `replacement_order` | per-article seeded shuffle; each level takes a prefix (nested levels) |
| `mix_text` / `mix_article` | swap chosen words for English by character offset; one row per level |
| `write_review_samples` | the 20-sample eyeball file |
| `romanise` / `same_name` | stretch: rough Latin spelling of a Hindi word; is the English the same name (similarity >= 0.55)? |
| `fact_words` / `stretch_rows` / `main_stretch` | stretch: replaceable fact words; targeted and random rows with equal counts -> `stretch_mixed.jsonl` |

**`prompts.py`**

| Name | Responsibility |
|---|---|
| `TASK` | shared instruction (English): 2-3 sentence summary in Hindi, Devanagari, summary only |
| `format_checklist` | checklist as bullet lines ("Keep these key facts ... exactly"), empty categories skipped |
| `build_messages` | one user message: task [+ checklist] + article |

**`generate.py`**

| Function | Responsibility |
|---|---|
| `set_seed` | seed Python, NumPy and torch |
| `load_model` | the chosen model (`config.MODELS`) in NF4 4-bit (float32 compute for `FLOAT32_MODELS`) with fp16 compute on CUDA; fp32 CPU/MPS fallback for code tests only |
| `summarise` | chat template (pinned date), greedy decode, token counts, `hit_max_tokens`, time |
| `drop_partial_line` / `load_done` | crash recovery: remove a half-written last line, collect finished keys |
| `jobs` / `main` | fixed job order (article, level, prompt); skip done; append + fsync each row; `--dry-run` / `--stretch` pick the input and output files; `--model <tag>` (not 1b) writes to `results/scale/<tag>/` and keeps only `SCALE_LEVELS`; stops if the first 6 summaries have no Devanagari |

**`score.py`**

| Function | Responsibility |
|---|---|
| `first_sentences` | drop a lead-in line ending in ":", keep the first 3 sentences |
| `match_key` / `contains` | normalise (Unicode, digits, nukta, case, punctuation) and match whole words |
| `numbers_in` | set of normalised numbers in a text |
| `recall` | share of checklist facts in the summary, overall and per type |
| `precision` | facts found in the summary (NER + regex), which are not in the original article |
| `score_all` | one score row per generation, plus length, compression, input English share |
| `metrics_table` | means per (level, prompt) -> `metrics.csv` |
| `without_latin` | precision ignoring "entities" with English letters -> `precision_no_latin` |
| `plot` | two panels (recall, precision), one line per prompt, shared 0-1 axis -> `chart.png` |
| `write_sheet` / `make_sheets` | blind 48-summary tagging sheet + key, 20-article checklist sheet (UTF-8 BOM, never overwrite) |
| `summarise_tags` | validate tags against the 6 categories, count per (level, prompt) |
| `english_form` / `recall_either_form` | stretch: a fact counts in its Hindi or English spelling |
| `score_stretch` | stretch scores and the condition x prompt table (paths from `config.model_paths`) |
| `paired_diffs` | a - b per article on any column (condition, level, prompt), mean and 95% bootstrap interval (seeded) |
| `add_f1` | per-summary F1 of recall and precision (NaN when the summary has no facts) |
| `plot_sizes` / `compare_sizes` | `--compare-sizes`: combine all models' scores -> `results/scale/size_*.csv` (incl. `size_mixing_diffs.csv`) and `size_chart.png` |

**`verify.py`**

| Function | Responsibility |
|---|---|
| `sentences` / `windows` / `best_windows` | split with the scorer's sentence rule; overlapping 4-sentence article windows; the 3 windows with most word overlap per summary sentence |
| `NLI` | mDeBERTa XNLI classifier on CUDA/MPS/CPU; returns entailment and contradiction probabilities |
| `check_summaries` | per summary: share of sentences entailed (`nli_support`), any sentence contradicted (`nli_contradicted`), lowest entailment |
| `auc` / `validate` | `--validate`: compare with the final hand tags (factual error, entity swap) -> `nli_validation.csv` |
| `run_all` / `summarise` | every model's summaries at levels 0/40/80 -> `nli_scores.jsonl`; tables `nli_metrics.csv` and `nli_diffs.csv` (`--summarise` rebuilds them without rerunning NLI) |

## Flow

1. `extract.py`: 8,847 test articles -> 6,414 pass the length filter -> 100 drawn with seed 42 -> checklist per article.
2. `mix.py`: per article, find eligible nouns and adjectives (about 75 per article), shuffle once, write 5 levels (level 0 = original text). Because only content words are eligible, level 80% makes about 19% of all words English (18.6% measured).
3. `tests/checks.py`: regex, digit normalisation, mixing percentage (within ±3 points), protection of facts, nesting.
4. `generate.py`: 100 articles x 5 levels x 2 prompts = 1,000 summaries, appended and fsynced one by one; on restart it drops any half-written line and skips finished (id, level, prompt) keys. Measured on the Kaggle T4: 6.5 s per summary, 1.8 h for all 1,000. About 45% of summaries hit the 200-token cap.
5. `score.py`: drop a lead-in line ending in ":", keep the first 3 sentences, then recall against the level-0 checklist and precision against the level-0 article; average per (level, prompt).

## Key data models

`data/checklists.jsonl`, one line per article:

```json
{"id": "...", "title": "...", "text": "<normalised article>", "reference_summary": "...", "n_words": 215,
 "checklist": {"entities": ["अंकारा", "..."], "numbers": ["130", "150000"], "dates": ["14 फरवरी"]},
 "protected_spans": [[start, end], ...]}
```

`data/mixed.jsonl`, one line per (article, level):

```json
{"id": "...", "level": 40, "text": "<mixed article>", "n_eligible": 65, "n_replaced": 26, "replaced": ["दावा", "..."]}
```

`results/generations.jsonl` (and `dry_run.jsonl`), one line per (article, level, prompt):

```json
{"id": "...", "level": 40, "prompt": "checklist", "summary": "...", "n_input_tokens": 649, "n_output_tokens": 136,
 "hit_max_tokens": false, "time_s": 7.49, "model": "meta-llama/Llama-3.2-1B-Instruct", "setup": "cuda-4bit"}
```

`results/metrics.csv`, one row per (level, prompt): `n, input_english_share, recall, recall_entities, recall_numbers, precision, precision_no_latin, n_without_facts, unsupported_per_summary, summary_words, summary_latin_share, compression, output_tokens_raw, hit_max_tokens, time_s`.

`data/stretch_mixed.jsonl`, one line per (article, condition): `{"id", "level": "targeted" | "random", "text", "n_replaced", "replaced", "fact_english": {Hindi word: English form}}`. `level` holds the condition so `generate.py` can treat these rows like the main ones.

`results/error_tags.csv`: `item, article_id, original_article, summary, auto_unsupported (hint, may be wrong), tags, notes`. Tags: entity swap, number drift, date shift, unsupported addition, language/script drift, none.

## External services and models

| Name | Used for | Notes |
|---|---|---|
| `csebuetnlp/xlsum` (HF dataset, `refs/convert/parquet`, `hindi/test/0000.parquet`) | articles | loading script broken on `datasets` 5.x, so Parquet is used |
| `ai4bharat/IndicNER` (HF model) | entities | **gated**: accept licence + `hf auth login` |
| Stanza `hi` models | POS tags, lemmas | downloaded automatically to `~/stanza_resources` |
| MUSE `hi-en.txt` (`dl.fbaipublicfiles.com`) | Hindi->English words | downloaded once to `data/dict/` |
| `wordfreq` (Python package) | ranking English candidates | offline |
| `meta-llama/Llama-3.2-1B-Instruct` (HF model) | summaries | **gated** (access granted), NF4 4-bit on a T4 |
| `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` (HF models) | model-size extension | **gated** (3.1 licence accepted 2026-11-10), NF4 4-bit on a T4 |
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (HF model) | NLI meaning check (`verify.py`) | open, runs locally on MPS/CPU; needs `sentencepiece` and `protobuf` |
| `google/gemma-2-2b-it`, `google/gemma-2-9b-it` (HF models) | second model family | **gated** (Gemma licence accepted 2026-11-11), NF4 4-bit with float32 compute on a T4 (`setup` = `cuda-4bit-fp32`) |

## Dependencies

See `requirements.txt` (tested with Python 3.12). Generation on the GPU also needs `bitsandbytes` and `accelerate`, installed in the Colab cell above; they do not run on macOS. Record the GPU versions printed by that cell in DECISIONS.md.

## Configuration

Everything is in `config.py`: seed, paths, dataset location, length filter, mixing levels, POS classes (nouns, adjectives), dictionary settings, NER model, threshold and entity stoplist, generation settings (model, 4-bit, fp16 compute, `MAX_NEW_TOKENS`=200, pinned `CHAT_DATE`, dry-run size and path), scoring settings (`SCORE_SENTENCES`, output paths, chart colours), and manual-check settings (tag levels, 8 articles, tag categories, sheet paths), and stretch settings (`STRETCH_NAME_SIMILARITY`, conditions, paths), NLI settings (`NLI_MODEL`, window size, top windows, threshold, paths), and model-size settings (`MODELS`, `SCALE_LEVELS`, `SCALE_DIR`, the `size_*` paths, `BOOTSTRAP_SAMPLES`, `FLOAT32_MODELS` (Gemma: float32 compute), and `model_paths(tag)`, which gives each model's input and output files). No secrets are stored in the repo: the Hugging Face token lives only in the local HF cache after `hf auth login` (on Colab/Kaggle, use their secrets feature, never paste it into a notebook cell).

## Repository and data licences

Public repository: https://github.com/Swapnil-Pramanik/FactIndic (branch `main`). The following are not committed (see `.gitignore`): `.venv/`, caches, `data/dict/` (the MUSE dictionary, which `mix.py` downloads), and `FactIndic_colab.zip` (rebuild it from `config.py`, `prompts.py`, `generate.py`, `requirements.txt` and the three `data/*.jsonl` files). Article text in `data/` and `results/` comes from XL-Sum (BBC Hindi), licensed CC BY-NC-SA 4.0, so this repository is for non-commercial academic use only. No licence file is included.
