"""All settings for the FactIndic experiment. Scripts import from here; nothing is hard-coded elsewhere."""

from pathlib import Path

# --- Reproducibility ---
SEED = 42

# --- Paths ---
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTICLE_IDS_PATH = DATA_DIR / "article_ids.txt"
CHECKLISTS_PATH = DATA_DIR / "checklists.jsonl"
MIXED_PATH = DATA_DIR / "mixed.jsonl"
TRANSLATIONS_PATH = DATA_DIR / "translations.json"
SAMPLES_PATH = DATA_DIR / "mixed_samples_for_review.txt"
GENERATIONS_PATH = RESULTS_DIR / "generations.jsonl"

# --- Data ---
DATASET_NAME = "csebuetnlp/xlsum"
# The dataset's loading script no longer runs on recent `datasets`, so we read
# the Parquet copy that Hugging Face auto-generates on this branch.
DATASET_REVISION = "refs/convert/parquet"
DATASET_FILE = "hindi/test/0000.parquet"
N_ARTICLES = 100
MIN_WORDS = 150
MAX_WORDS = 600

# --- Code-mixing ---
LEVELS = [0, 20, 40, 60, 80]  # percent of eligible content words replaced
STANZA_LANG = "hi"
# Only nouns and adjectives are replaced, not verbs (a deliberate change from PLAN.md).
# Real Hinglish brings English verbs in with a Hindi helper ("decide किया"); swapping a
# conjugated Hindi verb for a bare English one gives "उन्होंने said कि" or
# "prisoner created जाने", which nobody says.
CONTENT_POS = {"NOUN", "ADJ"}
# MUSE Hindi->English word list (Facebook Research), free to download.
DICT_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/hi-en.txt"
DICT_PATH = DATA_DIR / "dict" / "hi-en.txt"
# Among several dictionary translations we keep the most common English word
# (wordfreq Zipf scale, 0-8). Below this score the "translation" is usually a
# transliteration such as अदालत -> "adalat", so we treat the word as untranslatable.
MIN_ZIPF = 3.0

# --- Checklist extraction ---
NER_MODEL = "ai4bharat/IndicNER"  # gated: accept the licence on Hugging Face and log in
NER_LABELS = {"PER", "LOC", "ORG"}
NER_MIN_SCORE = 0.5
# NER occasionally tags a bare function word (e.g. के) as an entity. Such a "fact"
# appears in almost any summary and would inflate recall, so these are dropped.
ENTITY_STOPWORDS = {"के", "की", "का", "को", "में", "से", "ने", "पर", "है", "हैं", "और", "भी", "ये", "यह", "वह", "वे", "इस", "उस"}

# --- Generation (Day 2) ---
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"  # the main experiment's model; its results never change

# --- Model-size extension (after Day 5) ---
# The same pipeline on larger Llama models, to see whether the effects shrink with size.
# "1b" is the main experiment above; the others write under results/scale/<tag>/.
MODELS = {
    "1b": "meta-llama/Llama-3.2-1B-Instruct",
    "3b": "meta-llama/Llama-3.2-3B-Instruct",
    "8b": "meta-llama/Llama-3.1-8B-Instruct",  # a different release (3.1), same family
    # Second family (2026-11-11): Gemma 2, strong in Hindi, two sizes from ONE release, so
    # size is not mixed up with release as it is for Llama 8B.
    "gemma2b": "google/gemma-2-2b-it",   # gated: accept the Gemma licence on Hugging Face
    "gemma9b": "google/gemma-2-9b-it",
}
# Gemma 2 is known to overflow in float16 (it was trained in bfloat16, which the T4 lacks),
# so Gemma runs keep their unquantised weights and 4-bit compute in float32. Slower, same outputs
# otherwise; Llama keeps the original float16 compute.
FLOAT32_MODELS = {"gemma2b", "gemma9b"}
SCALE_LEVELS = [0, 40, 80]   # lean scope: 3 levels x 2 prompts + the stretch = 1,000 summaries per model
SCALE_DIR = RESULTS_DIR / "scale"
SIZE_METRICS_PATH = SCALE_DIR / "size_metrics.csv"            # level x prompt x model (levels 0/40/80)
SIZE_STRETCH_METRICS_PATH = SCALE_DIR / "size_stretch_metrics.csv"  # condition x prompt x model
SIZE_STRETCH_DIFFS_PATH = SCALE_DIR / "size_stretch_diffs.csv"      # targeted - random, paired, with 95% intervals
SIZE_MIXING_DIFFS_PATH = SCALE_DIR / "size_mixing_diffs.csv"      # L80 - L0 and checklist - plain, paired, 95% intervals
SIZE_CHART_PATH = SCALE_DIR / "size_chart.png"
BOOTSTRAP_SAMPLES = 2000


def model_paths(tag):
    """Input and output files for one model size; "1b" is the main experiment's files."""
    if tag == "1b":
        return {"generations": GENERATIONS_PATH, "scores": SCORES_PATH, "metrics": METRICS_PATH,
                "stretch_generations": STRETCH_GENERATIONS_PATH, "stretch_scores": STRETCH_SCORES_PATH,
                "stretch_metrics": STRETCH_METRICS_PATH}
    d = SCALE_DIR / tag
    return {"generations": d / "generations.jsonl", "scores": d / "scores.jsonl", "metrics": d / "metrics.csv",
            "stretch_generations": d / "stretch_generations.jsonl", "stretch_scores": d / "stretch_scores.jsonl",
            "stretch_metrics": d / "stretch_metrics.csv"}
LOAD_IN_4BIT = True       # NF4 weights via bitsandbytes; needs an NVIDIA GPU
COMPUTE_DTYPE = "float16"  # the T4 has no bfloat16 support
MAX_NEW_TOKENS = 200      # a 2-3 sentence Hindi summary needs ~80-150 tokens
PROMPTS = ["plain", "checklist"]
# Llama 3.2's chat template writes today's date into every prompt. Pinning it keeps
# the prompts identical no matter which day the script runs.
CHAT_DATE = "01 Nov 2026"
# Dry run: first N articles, written to a separate file so it never mixes with the real run.
DRY_RUN_ARTICLES = 10
DRY_RUN_PATH = RESULTS_DIR / "dry_run.jsonl"

# --- Scoring (Day 3) ---
# Summaries are cut to their first N sentences before scoring. About half of them
# ignore "2-3 sentences" and run into MAX_NEW_TOKENS; cutting enforces the prompt's
# own limit the same way for every summary (student's choice, see DECISIONS.md).
SCORE_SENTENCES = 3
SCORES_PATH = RESULTS_DIR / "scores.jsonl"   # one row per summary, with unsupported facts
METRICS_PATH = RESULTS_DIR / "metrics.csv"   # the level x prompt table
CHART_PATH = RESULTS_DIR / "chart.png"
# Chart colours: the first two slots of a colour-blind-checked categorical palette.
PROMPT_COLORS = {"plain": "#2a78d6", "checklist": "#eb6834"}

# --- Manual checks (Day 4) ---
TAG_LEVELS = [0, 40, 80]
TAG_ARTICLES = 8            # same 8 articles x 3 levels x 2 prompts = 48 summaries
TAG_CATEGORIES = ["entity swap", "number drift", "date shift", "unsupported addition",
                  "language/script drift", "none"]
ERROR_TAGS_PATH = RESULTS_DIR / "error_tags.csv"          # blind sheet the student fills in
ERROR_TAGS_KEY_PATH = RESULTS_DIR / "error_tags_key.csv"  # item -> level, prompt (kept apart)
ERROR_COUNTS_PATH = RESULTS_DIR / "error_tag_counts.csv"
CHECKLIST_CHECK_ARTICLES = 20
CHECKLIST_CHECK_PATH = RESULTS_DIR / "checklist_check.csv"

# --- Stretch: targeted vs random mixing (PLAN.md Day 5) ---
# Targeted replaces words inside facts (names, places, organisations) with their English
# form; random replaces the same number of non-fact words. A fact word is replaced only
# when its dictionary English is the Latin spelling of the same name (पाकिस्तान -> Pakistan),
# never a literal translation of a name (मुलायम -> "soft", अरब -> "billion").
STRETCH_NAME_SIMILARITY = 0.55   # similarity of rough romanisation vs English, 0-1
STRETCH_CONDITIONS = ["random", "targeted"]
STRETCH_MIXED_PATH = DATA_DIR / "stretch_mixed.jsonl"
STRETCH_GENERATIONS_PATH = RESULTS_DIR / "stretch_generations.jsonl"
STRETCH_SCORES_PATH = RESULTS_DIR / "stretch_scores.jsonl"
STRETCH_METRICS_PATH = RESULTS_DIR / "stretch_metrics.csv"

# --- Automatic meaning check (verify.py, 2026-11-11) ---
# Sentence-level NLI: is each summary sentence entailed by the original article?
# A multilingual NLI model trained on XNLI (which includes Hindi) and machine-translated MNLI.
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_WINDOW = 4          # article sentences per premise window (windows overlap by one sentence)
NLI_TOP_WINDOWS = 3     # windows with the most word overlap that are checked per summary sentence
NLI_THRESHOLD = 0.5     # probability needed to call a sentence entailed / contradicted
NLI_MAX_TOKENS = 512
NLI_BATCH = 16
NLI_VALIDATION_PATH = RESULTS_DIR / "nli_validation.csv"   # NLI vs the 48 hand tags
NLI_SCORES_PATH = RESULTS_DIR / "nli_scores.jsonl"         # one row per summary, all models, levels 0/40/80
NLI_METRICS_PATH = RESULTS_DIR / "nli_metrics.csv"         # model x level x prompt
NLI_DIFFS_PATH = RESULTS_DIR / "nli_diffs.csv"             # L80 - L0 and checklist - plain, paired, 95% intervals

# Content preservation against the human-written XL-Sum reference summary (content.py).
CONTENT_BERT_MODEL = "xlm-roberta-large"   # BERTScore's recommended multilingual model (layer 17 by default)
CONTENT_BATCH = 32
CONTENT_SCORES_PATH = RESULTS_DIR / "content_scores.jsonl"   # one row per summary: ROUGE-1/2/L and BERTScore F1
CONTENT_METRICS_PATH = RESULTS_DIR / "content_metrics.csv"   # model x run x level x prompt, plus Lead-1/Lead-3
CONTENT_DIFFS_PATH = RESULTS_DIR / "content_diffs.csv"       # L80 - L0, targeted - random, checklist - plain
