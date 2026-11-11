# Future work

Ideas that came up during the project but are outside PLAN.md's scope (section 10). Not built.

- **Natural verb mixing.** Mix verbs the way Hinglish does, as "English + Hindi helper" ("कहा" -> "say किया"), instead of leaving all verbs in Hindi. Needs tense/gender/number handling.
- **Context-aware word translation.** Replace the context-free MUSE lookup (हवाई -> "hawaii", विराम -> "comma") with a translation that sees the sentence, e.g. an MT model or an LLM prompted per sentence.
- **Real code-mixed text.** Repeat the study on naturally written Hinglish news or social media instead of machine-mixed text.
- **Stricter length control.** 40-53% of summaries ignored "2-3 sentences". Try a stricter prompt, a stop rule, or a few-shot example, instead of cutting at scoring time.
- **Number words in the checklist.** Recognise तीन, चार, लाख, करोड़ and dates like "तीन फ़रवरी" (28 of the 39 facts the checklist missed in the 20-article check).
- **Fuzzier fact matching.** Accept spelling variants (अमेरिका / अमरीका) and inflected forms when scoring, or use the NLI checker the plan listed as optional.
- **Bigger sample for hand tagging.** 8 summaries per cell is too few to compare the two prompts; a larger, purely human-tagged sample (ideally two taggers) would allow that.
- **Targeted vs random mixing** (PLAN.md Day 5 stretch): done; see REPORT.md section 6. Follow-ups: score names by fuzzy or transliteration-aware matching so respellings (इरान vs ईरान) are separated from wrong names; check a sample of the respellings by hand; translate names the similarity rule rejects (भारत -> India) with a hand-made list.
- **More models.** The size extension (REPORT.md section 7) covers Llama 1B/3B/8B at levels 0/40/80. Open: a 3.1-release 1B/3B or a 3.2-release 8B equivalent to separate size from release, all five levels, and other model families. Other languages (Marathi, Romanized Hinglish) remain out of scope.
