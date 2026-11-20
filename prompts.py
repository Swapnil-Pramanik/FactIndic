"""All prompt text lives in this file.

Prompting ladder (each step adds one thing to the step before):
  plain       task only
  checklist   + the article's fact checklist, "keep exactly as written"
  instructed  + explicit factuality rules from the course brief (no unsupported information,
                exact names/numbers/dates, correct roles, how to read code-mixed words)
  verified    instructed draft, then a second pass in which the same model checks the draft
              against the article and removes or corrects unsupported details
The instructed and verified prompts were written and frozen before any of their outputs
were seen (see DECISIONS.md); they were not tuned on the 100 test articles.
"""

# The instructions are in English because a 1B model follows English instructions
# far more reliably than Hindi ones; the summary itself is requested in Hindi.
TASK = (
    "Summarise the following news article in 2-3 sentences. "
    "Write the summary in Hindi, in Devanagari script. Output only the summary."
)

CHECKLIST_INTRO = "Keep these key facts from the article exactly as they are written:"

CHECKLIST_LABELS = {
    "entities": "Names, places and organisations",
    "numbers": "Numbers",
    "dates": "Dates",
}

# The brief's list for a factuality-aware prompt: concise, preserve facts, no unsupported
# information, preserve entities, numbers and dates, handle code-mixed expressions.
# Rule 4 targets the most common hand-tagged error (role swaps), rule 2 the name respellings.
FACTUALITY_RULES = """Follow these rules:
1. Use only information that is stated in the article. Do not add any fact, opinion, cause or background that the article does not contain.
2. Write every name of a person, place or organisation exactly as it is spelled in the article. Do not re-spell or transliterate it differently.
3. Copy numbers and dates exactly. Do not round, convert or estimate them.
4. Keep who did what to whom exactly as in the article. Do not swap the roles of people, groups or countries.
5. The article may mix Hindi with some English words. Read each English word with its normal meaning and write it in Hindi in the summary.
6. Include the most important facts: the main event, who was involved, and the key numbers."""

VERIFY_TASK = """Below are a news article and a draft summary of it in Hindi. Check the draft sentence by sentence against the article:
1. Remove any detail that the article does not state.
2. Correct any name, number or date that differs from the article, using the article's spelling.
3. Correct any sentence in which people, groups or countries have the wrong roles.
Keep the summary to 2-3 sentences, in Hindi, in Devanagari script. If the draft is already correct, repeat it unchanged. Output only the final summary, with no explanation."""

# Single-pass prompts that build_messages accepts; "verified" is built from an instructed draft.
SINGLE_PASS = ("plain", "checklist", "instructed")


def format_checklist(checklist):
    """Turn a checklist dict into bullet lines, skipping empty categories."""
    lines = [CHECKLIST_INTRO]
    for key, label in CHECKLIST_LABELS.items():
        if checklist[key]:
            lines.append(f"- {label}: {', '.join(checklist[key])}")
    return "\n".join(lines)


def build_messages(prompt, article_text, checklist):
    """Chat messages for one generation; `prompt` is "plain", "checklist" or "instructed"."""
    parts = [TASK]
    if prompt == "instructed":
        parts.append(FACTUALITY_RULES)
    if prompt in ("checklist", "instructed"):
        parts.append(format_checklist(checklist))
    elif prompt != "plain":
        raise ValueError(f"unknown prompt: {prompt}")
    parts.append(f"Article:\n{article_text}")
    return [{"role": "user", "content": "\n\n".join(parts)}]


def build_verify_messages(article_text, draft):
    """Second pass of the `verified` prompt: check the (3-sentence) draft against the same article."""
    content = f"{VERIFY_TASK}\n\nArticle:\n{article_text}\n\nDraft summary:\n{draft}"
    return [{"role": "user", "content": content}]
