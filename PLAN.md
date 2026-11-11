# FactIndic: Implementation Plan (5-day student scope)

You are helping a college student build a small, finished experiment in 4-5 days. Follow this plan exactly. When something is unspecified, pick the simplest option, write the choice in `DECISIONS.md`, and keep going. Do not expand scope.

## 1. Question

As Hindi news text becomes more code-mixed with English, do LLM summaries lose facts or gain wrong ones? Does adding an automatically extracted fact checklist to the prompt help?

No model training. Inference only.

## 2. Fixed design (do not change)

| Item | Value |
|---|---|
| Data | Hindi split of XL-Sum (`csebuetnlp/xlsum`, config `hindi`), first 100 test articles that pass a length filter (e.g. 150-600 words). Seeded sampling. |
| Mixing levels | 0, 20, 40, 60, 80 (%) |
| Prompts | `plain` vs `checklist` |
| Model | One open 2-4B instruct model with decent Hindi, loaded 4-bit. Check what loads on a free T4 (candidates: Qwen2.5-3B-Instruct, Gemma 2B/4B-it, Llama-3.2-3B-Instruct). Pick one, record it, never switch. |
| Decoding | temperature 0 (greedy), fixed max_new_tokens, seed set everywhere |
| Total generations | 100 articles x 5 levels x 2 prompts = 1,000 |
| Hardware | free Colab/Kaggle T4. Scripts must resume after a crash, so write each result to disk as it finishes. |

## 3. Code-mixing definition

- Base language is Hindi (Devanagari). English words are inserted in Latin script, e.g. "सरकार ने नई policy की घोषणा की".
- Level L% means: L% of eligible content words in the article (nouns, adjectives, verbs) are replaced by their English equivalent. Function words (ने, की, है, ...) are never replaced.
- Named entities, numbers, dates and places are never replaced in the main experiment. They are what the checklist tracks, so replacing them would change what we measure.
- Choose replaced words with a seeded random sample, so levels are nested (every word replaced at 20% is also replaced at 40%, and so on).
- Getting the English word: simplest working option. Candidates are an English-Hindi dictionary, or the same LLM prompted to give a one-word English equivalent for each selected word. Cache translations to disk.
- Machine-mixed text is not natural. State this in the final report.

## 4. Fact checklist

Extract automatically from the ORIGINAL (level 0) article, so the checklist is identical across levels:
- numbers and dates: regex, including Devanagari digits (१२ = 12) and common date formats
- names, places, organisations: an off-the-shelf Hindi NER model (check availability, e.g. `ai4bharat/IndicNER`) or the simplest workable fallback
- store as JSON per article: `{"entities": [...], "numbers": [...], "dates": [...]}`
- known weakness: extraction is imperfect. Report its rough quality from a 20-article eyeball check.

In the `checklist` prompt the checklist is listed, with the instruction to keep these facts exactly. Both prompts ask for a 2-3 sentence summary in Hindi of the given (possibly mixed) article. Keep prompt text in one file, `prompts.py`.

## 5. Scoring

Primary (required):
- **Recall**: share of checklist facts present in the summary (after normalisation: digit script, whitespace, Latin/Devanagari forms of the same name where simple).
- **Precision**: share of entities/numbers/dates found in the summary that also appear in the ORIGINAL article. Facts that don't appear are flagged as unsupported.
- Also report average summary length (tokens), compression ratio (summary length / article length), and generation time.

Secondary (only if Day 3 has time): sentence-level multilingual NLI (article sentences vs each summary sentence). Report separately, never mixed into the primary numbers.

Manual error tagging (Day 4): hand-tag 40-60 summaries drawn evenly from levels 0/40/80 and both prompts. Categories: entity swap, number drift, date shift, unsupported addition, language/script drift, none. Save as `results/error_tags.csv`.

Output: one results table (level x prompt) with recall, precision, length, and one chart (recall and precision against mixing level, two lines per metric for the two prompts).

## 6. Repo layout

```
factindic/
  PLAN.md
  DECISIONS.md          # every choice you made, one line each
  config.py             # all settings in one place
  prompts.py
  extract.py            # load data, build checklists  -> data/checklists.jsonl
  mix.py                # code-mixing                   -> data/mixed.jsonl
  generate.py           # run model, resumable          -> results/generations.jsonl
  score.py              # metrics, table, chart         -> results/metrics.csv, results/chart.png
  tests/checks.py       # about 10 asserts (see section 8)
  README.md             # how to run, in order
```

Four scripts at most for the pipeline. Plain functions, no classes, no packaging.

## 7. Day plan and acceptance criteria

**Day 1: data, extractor, mixer**
- Load and filter the data, save the 100 article ids.
- `extract.py` and `mix.py` working. Run `tests/checks.py` (all pass).
- Print 20 mixed samples across levels to a text file for the student to eyeball. Pause and ask the student to review them before generating anything.
- Done when: checklists and mixed articles exist for all 100 articles x 5 levels.

**Day 2: generation**
- `generate.py` with resume support. First run 10 articles end to end as a dry run, show the outputs, then run the full 1,000.
- Done when: `generations.jsonl` has 1,000 rows with article id, level, prompt, summary, time.

**Day 3: scoring**
- `score.py` produces the metrics table and chart. Add the NLI checker only if there is time.
- Done when: `metrics.csv` and `chart.png` exist and the numbers can be reproduced by rerunning the script.

**Day 4: manual tagging and write-up**
- Produce the tagging sheet (CSV with article, summary, empty tag column) for the student to fill in, then summarise tag counts per level and prompt.
- Draft a short report: question, method, results table, chart, error counts, limitations.

**Day 5: buffer**
- Fix bugs, polish README, prepare slides.
- Stretch only if everything above is done: targeted vs random mixing at equal mixing level (replace fact words vs non-fact words).

## 8. Tests (minimal, student-scale)

`tests/checks.py`, plain `assert` statements, about 10 total:
- extractor finds the numbers and dates in 3-4 hand-written Hindi sentences
- Devanagari and Western digits normalise to the same value
- mixer replaces the target percentage of eligible words within +/-3 points on a sample
- mixer never changes entities, numbers or dates
- levels are nested (words replaced at 20% are a subset of those at 40%)
- scorer gives recall 1.0 for a summary containing the full checklist and 0.0 for an empty one

No test framework, coverage target or CI. Do not unit-test LLM output.

## 9. Code style

- Readable over clever. Short functions with a one-line docstring.
- Settings only in `config.py`, never hard-coded in scripts.
- Save intermediate results as JSONL/CSV so any stage can rerun alone.
- Prefer existing libraries over custom code.
- Add comments where a choice might surprise the student. They must be able to explain every line.

## 10. Out of scope (do NOT build)

Training or fine-tuning, Marathi, Romanized Hinglish, more than one model, model-size sweeps, hyperparameter tuning, poisoned-checklist test, native vs English-pivot verification, verify-and-patch, significance testing beyond reporting counts and averages, web apps, packaging, CI. If an idea comes up, append it to `FUTURE_WORK.md` and move on.

## 11. How to work with the student

- Before each day's work, state in two lines what you will do.
- Stop and ask only for: the Day 1 eyeball review, picking the model if two are close, and anything that would change a number in section 2.
- If a library or model name above doesn't exist or won't install, choose the closest alternative, log it in `DECISIONS.md`, and continue.
- Do not claim results you haven't run. Every number in the report must come from `metrics.csv`.
