"""Generate a summary for every (article, level, prompt), resumably -> results/generations.jsonl.

Usage:  python generate.py            # full run, 100 articles x 5 levels x 2 prompts
        python generate.py --dry-run  # first DRY_RUN_ARTICLES articles, to results/dry_run.jsonl
        python generate.py --stretch  # targeted vs random mixing, 100 x 2 x 2 -> results/stretch_generations.jsonl
        python generate.py --model 3b [--stretch]  # larger model, levels SCALE_LEVELS only -> results/scale/3b/
Rerunning either command skips rows already on disk, so a crashed run just continues.
"""

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from prompts import build_messages


def set_seed(seed):
    """Seed every random number generator we touch (greedy decoding is deterministic anyway)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(model_name, float32=False):
    """Load the tokenizer and the model; 4-bit on a CUDA GPU as the plan requires.
    float32=True (Gemma) computes in float32 instead of COMPUTE_DTYPE, to avoid float16 overflow."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if torch.cuda.is_available() and config.LOAD_IN_4BIT:
        compute = torch.float32 if float32 else getattr(torch, config.COMPUTE_DTYPE)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute)
        extra = {"dtype": torch.float32} if float32 else {}
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant,
                                                     device_map="cuda", **extra)
        setup = "cuda-4bit-fp32" if float32 else "cuda-4bit"
    else:
        # Only for testing the code on a laptop: full precision on CPU/Apple GPU.
        # These outputs differ from the T4 4-bit run and must never be reported.
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
        setup = f"{device}-fp32"
    model.eval()
    return tokenizer, model, setup


def summarise(tokenizer, model, messages):
    """Greedy-decode one summary; return the text, token counts and seconds taken."""
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        date_string=config.CHAT_DATE,
    )["input_ids"].to(model.device)
    start = time.time()
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                                temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id)
    seconds = time.time() - start
    new_tokens = output[0, input_ids.shape[1]:]
    return {
        "summary": tokenizer.decode(new_tokens, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False).strip(),
        "n_input_tokens": input_ids.shape[1],
        "n_output_tokens": len(new_tokens),
        # True when the model was cut off at MAX_NEW_TOKENS (often a repetition loop).
        "hit_max_tokens": len(new_tokens) >= config.MAX_NEW_TOKENS,
        "time_s": round(seconds, 2),
    }


def drop_partial_line(path):
    """Cut off a half-written last line left by a crash, so new rows don't get glued onto it."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def load_done(path):
    """Keys (id, level, prompt) of rows already written, so we can skip them."""
    if not path.exists():
        return set()
    drop_partial_line(path)
    return {(r["id"], r["level"], r["prompt"]) for r in map(json.loads, open(path, encoding="utf-8"))}


def recent_summaries(path, n):
    """The last n summaries written to path."""
    return [json.loads(line)["summary"] for line in open(path, encoding="utf-8").readlines()[-n:]]


def jobs(mixed_path, n_articles=None, levels=None):
    """Every (mixed row, prompt, checklist) to generate, in a fixed order; `levels` keeps only those levels."""
    checklists = {a["id"]: a["checklist"] for a in map(json.loads, open(config.CHECKLISTS_PATH, encoding="utf-8"))}
    ids = list(checklists)[:n_articles]
    rows = [r for r in map(json.loads, open(mixed_path, encoding="utf-8"))
            if r["id"] in ids and (levels is None or r["level"] in levels)]
    for row in rows:
        for prompt in config.PROMPTS:
            yield row, prompt, checklists[row["id"]]


def main():
    """Run every job not yet on disk, appending each result as soon as it finishes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="first DRY_RUN_ARTICLES articles only")
    parser.add_argument("--stretch", action="store_true", help="targeted vs random mixing (Day 5 stretch)")
    parser.add_argument("--model", choices=list(config.MODELS), default="1b",
                        help="model size; anything but 1b is the model-size extension (SCALE_LEVELS only)")
    args = parser.parse_args()
    model_name = config.MODELS[args.model]
    if args.stretch:
        # Stretch rows hold the condition ("targeted"/"random") in "level", so everything below is shared.
        mixed_path, out_path, n_articles = config.STRETCH_MIXED_PATH, config.STRETCH_GENERATIONS_PATH, None
    elif args.dry_run:
        mixed_path, out_path, n_articles = config.MIXED_PATH, config.DRY_RUN_PATH, config.DRY_RUN_ARTICLES
    else:
        mixed_path, out_path, n_articles = config.MIXED_PATH, config.GENERATIONS_PATH, None
    levels = None
    if args.model != "1b":
        # Larger models: own folder, so the main 1B results are never touched.
        out_path = config.SCALE_DIR / args.model / out_path.name
        levels = None if args.stretch else config.SCALE_LEVELS
    todo = list(jobs(mixed_path, n_articles, levels))
    done = load_done(out_path)
    todo = [j for j in todo if (j[0]["id"], j[0]["level"], j[1]) not in done]
    print(f"{len(done)} rows already done, {len(todo)} to go -> {out_path}")
    if not todo:
        return

    set_seed(config.SEED)
    tokenizer, model, setup = load_model(model_name, float32=args.model in config.FLOAT32_MODELS)
    if not setup.startswith("cuda-4bit") and not args.dry_run:
        raise SystemExit("The full run must use the T4 with 4-bit loading. Use --dry-run to test locally.")
    print(f"model {model_name} on {setup}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, (row, prompt, checklist) in enumerate(todo, 1):
            result = summarise(tokenizer, model, build_messages(prompt, row["text"], checklist))
            record = {"id": row["id"], "level": row["level"], "prompt": prompt, **result,
                      "model": model_name, "setup": setup}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Flush to disk now, so a crash loses at most the row being generated.
            f.flush()
            os.fsync(f.fileno())
            print(f"[{i}/{len(todo)}] {row['id']} L{row['level']} {prompt}: "
                  f"{result['n_output_tokens']} tok, {result['time_s']}s")
            # Stop early if a new model setup produces broken text (e.g. float16 overflow gives
            # empty or non-Hindi output), instead of wasting hours of GPU time.
            if i == 6 and not any(re.search(r"[\u0900-\u097F]", r) for r in recent_summaries(out_path, 6)):
                raise SystemExit("The first 6 summaries contain no Devanagari: the model output looks broken.")


if __name__ == "__main__":
    main()
