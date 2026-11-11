"""The two prompts, plain and checklist. All prompt text lives in this file."""

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


def format_checklist(checklist):
    """Turn a checklist dict into bullet lines, skipping empty categories."""
    lines = [CHECKLIST_INTRO]
    for key, label in CHECKLIST_LABELS.items():
        if checklist[key]:
            lines.append(f"- {label}: {', '.join(checklist[key])}")
    return "\n".join(lines)


def build_messages(prompt, article_text, checklist):
    """Chat messages for one generation; `prompt` is "plain" or "checklist"."""
    parts = [TASK]
    if prompt == "checklist":
        parts.append(format_checklist(checklist))
    elif prompt != "plain":
        raise ValueError(f"unknown prompt: {prompt}")
    parts.append(f"Article:\n{article_text}")
    return [{"role": "user", "content": "\n\n".join(parts)}]
