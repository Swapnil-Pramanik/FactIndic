// Builds the IEEE-style conference paper (paper/FactIndic_IEEE.docx) from the project's results.
// Every number below is copied from REPORT.md / results/*.csv; keep them in sync if results change.
// Usage: node paper/build_paper.js   (needs the `docx` npm package)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType,
  WidthType, BorderStyle, SectionType, ExternalHyperlink, VerticalAlign,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const FONT = "Times New Roman";
const PAGE = { size: { width: 12240, height: 15840 }, margin: { top: 1080, bottom: 1440, left: 900, right: 900 } };
const COL = 5040; // one column, in DXA (3.5 in)

// ---------- text helpers ----------
// Inline markup: *italic*, **bold**, ^sup^ are not needed; keep plain runs, italic via {i: true}.
function runs(text, opts = {}) {
  // Split on _..._ for italics.
  const parts = text.split(/(_[^_]+_)/g).filter(Boolean);
  return parts.map((p) => p.startsWith("_") && p.endsWith("_")
    ? new TextRun({ text: p.slice(1, -1), italics: true, font: FONT, size: opts.size || 20, bold: opts.bold })
    : new TextRun({ text: p, font: FONT, size: opts.size || 20, bold: opts.bold, italics: opts.italics }));
}
const P = (text, o = {}) => new Paragraph({
  children: runs(text, o), alignment: o.align || AlignmentType.JUSTIFIED,
  indent: o.noIndent ? undefined : { firstLine: 202 }, spacing: { after: o.after || 0, before: o.before || 0, line: 228 },
});
const H1 = (num, text) => new Paragraph({
  children: [new TextRun({ text: `${num}. `, font: FONT, size: 20 }), new TextRun({ text, font: FONT, size: 20, smallCaps: true })],
  alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
});
const H2 = (letter, text) => new Paragraph({
  children: [new TextRun({ text: `${letter}. ${text}`, font: FONT, size: 20, italics: true })],
  spacing: { before: 120, after: 60 }, keepNext: true,
});
const BULLET = (label, text) => new Paragraph({
  children: [new TextRun({ text: label + "\t", font: FONT, size: 20 }), ...runs(text)],
  alignment: AlignmentType.JUSTIFIED, indent: { left: 288, hanging: 200 }, spacing: { after: 20, line: 228 },
  tabStops: [{ type: "left", position: 288 }],
});

// ---------- tables ----------
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(caption, title, header, rows, widths, note) {
  const total = widths.reduce((a, b) => a + b, 0);
  const cell = (t, i, isHead, rowIdx, nRows) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 20, bottom: 20, left: 50, right: 50 },
    borders: {
      top: isHead ? thin : none, left: none, right: none,
      bottom: isHead || rowIdx === nRows - 1 ? thin : none,
    },
    children: [new Paragraph({
      alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: [new TextRun({ text: t, font: FONT, size: 15, bold: isHead })],
    })],
  });
  const out = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new TextRun({ text: caption, font: FONT, size: 16, smallCaps: true })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, keepNext: true,
      children: [new TextRun({ text: title, font: FONT, size: 16, smallCaps: true })] }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths, alignment: AlignmentType.CENTER,
      rows: [
        new TableRow({ tableHeader: true, children: header.map((h, i) => cell(h, i, true, -1, rows.length)) }),
        ...rows.map((r, ri) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false, ri, rows.length)) })),
      ],
    }),
  ];
  if (note) out.push(new Paragraph({ spacing: { before: 40, after: 120 }, alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text: note, font: FONT, size: 14 })] }));
  else out.push(new Paragraph({ spacing: { after: 80 }, children: [] }));
  return out;
}

// ---------- figures ----------
function figure(file, pxW, pxH, caption) {
  const w = 336; // 3.5 in at 96 px/in
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true, children: [
      new ImageRun({ type: "png", data: fs.readFileSync(path.join(ROOT, file)),
        transformation: { width: w, height: Math.round((w * pxH) / pxW) },
        altText: { title: caption, description: caption, name: path.basename(file) } }),
    ] }),
    new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { before: 60, after: 160 }, children: [
      new TextRun({ text: caption.split(".")[0] + ". ", font: FONT, size: 16 }),
      new TextRun({ text: caption.split(".").slice(1).join(".").trim(), font: FONT, size: 16 }),
    ] }),
  ];
}

// ---------- front matter (single column) ----------
const front = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [
    new TextRun({ text: "Do Code-Mixed Inputs Make LLM Summaries Less Faithful? Factuality-Aware Summarization of Hindi–English News", font: FONT, size: 44 }),
  ] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Swapnil Pramanik", font: FONT, size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "CSET 346 Natural Language Processing", font: FONT, size: 20, italics: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 300 }, children: [new TextRun({ text: "Bennett University, Greater Noida, India", font: FONT, size: 20, italics: true })] }),
];

// ---------- body (two columns) ----------
const abstract = new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 100, line: 220 }, children: [
  new TextRun({ text: "Abstract", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—Large language models (LLMs) write fluent summaries, but fluency does not guarantee factual consistency, and much Indian news and social media text mixes Hindi with English. We ask whether code-mixed input makes LLM summaries lose or invent facts, and whether factuality-aware prompting helps. We take 100 BBC Hindi articles from XL-Sum and mix them with English at five controlled levels (0–80% of eligible nouns and adjectives, up to 18.6% of all words). Five open models (Llama 3.2 1B and 3B, Llama 3.1 8B, Gemma 2 2B and 9B) summarize them with up to four prompts, 6,600 summaries in all. We measure content preservation (fact recall, ROUGE, BERTScore), meaning preservation (sentence-level natural language inference, validated against human error tags) and factual consistency (fact precision, unsupported facts), with paired bootstrap confidence intervals. Mixing does not make models drop facts, but it raises unsupported facts per summary for all three Llama models (Llama 1B: 0.16 to 0.38) and doubles the share of summaries with a contradicted sentence for the 1B model. Switching the fact words themselves to English hurts most. A fact-checklist prompt raises recall for every model; adding explicit factuality rules gives the best result (Gemma 2 9B, F1 0.59) and removes the effect of mixing on unsupported facts for that model, while a self-verification pass by the same model lowers recall without removing errors.", font: FONT, size: 18, bold: true }),
] });
const keywords = new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 220 }, children: [
  new TextRun({ text: "Index Terms", font: FONT, size: 18, bold: true, italics: true }),
  new TextRun({ text: "—abstractive summarization, factual consistency, hallucination, code-mixing, Hindi, large language models, prompting.", font: FONT, size: 18, bold: true }),
] });

const body = [
  abstract, keywords,

  H1("I", "Introduction"),
  P("Automatic summarization is increasingly applied to news, social media and government text in Indian languages. Modern LLMs produce summaries that read well, but a fluent summary can still change a name, a number or who did what, or add a detail the source never stated [1], [2]. Such errors are hard to spot for a reader who has not seen the source, which is exactly the reader a summary is written for."),
  P("The problem is sharper for Indic text because much of it is code-mixed: speakers switch between Hindi and English within a sentence, as in “भारत में pollution का level लगातार increase हो रहा है”. Code-mixing is a known challenge for NLP models [3], [4], yet faithfulness studies of summarization are almost always run on monolingual text. We therefore ask two questions. First, as Hindi news becomes more code-mixed, do LLM summaries lose facts or gain wrong ones? Second, can factuality-aware prompting reduce these errors?"),
  P("To answer them without the confounds of comparing different documents, we build controlled code-mixed versions of the _same_ articles. Our contributions are:"),
  BULLET("•", "a controlled code-mixing procedure that replaces 0–80% of eligible Hindi nouns and adjectives with English while protecting the facts to be checked, plus a targeted variant that switches the fact words themselves;"),
  BULLET("•", "an evaluation of five open LLMs from two families on 6,600 summaries, covering the three objectives of content, meaning and factual preservation, with paired bootstrap intervals;"),
  BULLET("•", "a comparison of four prompting strategies, from a plain prompt to a fact checklist, explicit factuality rules and self-verification;"),
  BULLET("•", "a human error analysis showing that role swaps, which string metrics cannot see, are the most common error, and an NLI check validated against it."),

  H1("II", "Related Work"),
  P("_Factuality in summarization._ Maynez et al. [1] showed that abstractive summarizers frequently hallucinate content not supported by the source. FactCC [2] detects such inconsistencies with a classifier trained on synthetic errors."),
  P("_Code-mixing._ GLUECoS [3] and LinCE [4] benchmark models on code-switched tasks such as language identification, tagging and sentiment, but not on summarization faithfulness. We do not use naturally code-mixed text; instead we generate graded mixing from monolingual news, which lets us measure the effect of the amount of mixing on otherwise identical input."),
  P("_NLI-based checking._ SummaC [5] detects inconsistent summaries with natural language inference (NLI) between source and summary sentences. We use a sentence-level NLI check in the same spirit, together with simpler, interpretable fact matching."),
  P("_Hindi summarization and LLMs._ XL-Sum [6] provides BBC news articles and one-line reference summaries in 44 languages, including Hindi. A recent Hindi LLM benchmark [7] ranked Gemma-2-9b-it the most capable small open model, which motivated our choice of the Gemma 2 family alongside Llama."),

  H1("III", "Data and Controlled Code-Mixing"),
  H2("A", "Articles and fact checklists"),
  P("We use the Hindi test split of XL-Sum [6], keep articles of 150–600 words (6,414 of 8,847) and draw 100 with a fixed seed. For each article we build a _fact checklist_: names of people, places and organizations found by IndicNER [8], and numbers and dates found by regular expressions (Western and Devanagari digits). A checklist holds on average 13.5 entities, 2.5 numbers and 0.3 dates. A manual check of 20 articles found 95% of checklist items correct and the checklist covering about 90% of the facts in an article; most missed facts are numbers written as words."),
  H2("B", "Code-mixing procedure"),
  P("At level _L_, _L_% of the eligible words of an article are replaced by an English word in Latin script. Eligible words are nouns and adjectives (Stanza POS tags [9]) that are not part of any checklist fact and have a translation in the MUSE Hindi–English dictionary [10]; there are about 75 per article. Verbs are never replaced, because natural Hinglish brings English verbs in with a Hindi helper (“decide किया”) and a bare English verb in place of a conjugated Hindi one reads unnaturally. Levels 0, 20, 40, 60 and 80 are nested: every word replaced at 20% is also replaced at 40%. Because only content words are eligible, the share of all words that are English is 4.7%, 9.3%, 14.0% and 18.6% at levels 20 to 80."),
  H2("C", "Targeted mixing"),
  P("The main procedure never touches checklist facts. To test the effect of English fact words, as in real Hinglish (“Twitter”, “BJP”), each article also gets two versions with the same number of English words (11.4 on average): _targeted_, where words inside checklist facts are replaced by their English name, and _random_, where the same number of non-fact words are replaced."),

  H1("IV", "Models, Prompts and Evaluation"),
  H2("A", "Models"),
  P("We use five instruction-tuned open models: Llama 3.2 1B (the main model) and 3B, Llama 3.1 8B [11], and Gemma 2 2B and 9B [12]. All run in 4-bit NF4 quantization [13] with greedy decoding and at most 200 new tokens on an NVIDIA T4 (Kaggle). Llama 1B summarizes all five levels; the other models use levels 0, 40 and 80 plus the targeted-mixing pairs. A rerun of 100 summaries reproduced them word for word."),
  H2("B", "Prompts"),
  P("All prompts are in English and ask for a 2–3 sentence summary in Hindi. We compare four strategies, each adding one element to the previous one:"),
  BULLET("•", "_plain:_ the task and the article;"),
  BULLET("•", "_checklist:_ also lists the checklist facts, to be kept “exactly as they are written”;"),
  BULLET("•", "_instructed:_ also gives six rules: use only the article’s information; spell names as the article does; copy numbers and dates exactly; keep who did what to whom; read English words with their normal meaning and write in Hindi; include the main event, people and key numbers;"),
  BULLET("•", "_verified:_ the instructed summary is given back to the same model with the article, which must remove unsupported details and correct names, numbers, dates and roles."),
  P("Plain and checklist were run on all models; instructed and verified on the weakest (Llama 1B) and strongest (Gemma 2 9B) model at levels 0, 40 and 80. The last two prompts were written from the task brief and the earlier error analysis and frozen before any output was generated."),
  H2("C", "Metrics"),
  P("Every summary is cut to its first three sentences before scoring, since smaller models often ignore the length instruction. _Factual consistency_ is measured by fact precision, the share of entities, numbers and dates in the summary that appear in the original article, and by the number of unsupported facts per summary. _Content preservation_ is measured by fact recall, the share of checklist facts found in the summary, and by ROUGE-1/2/L [14] and BERTScore [15] (xlm-roberta-large [16]) against the XL-Sum reference summary. F1 is the per-summary harmonic mean of recall and precision. _Meaning preservation_ is measured by NLI: each summary sentence is checked against the three most similar four-sentence windows of the article with a multilingual mDeBERTa model [17], [18] trained on XNLI [19]; a sentence is supported if any window entails it with probability at least 0.5, and contradicted if it is not supported and the closest window contradicts it. We also report summary length, compression ratio (summary words / article words) and inference time per summary."),
  H2("D", "Statistics and human checks"),
  P("Differences between conditions are computed per article and given with paired percentile bootstrap 95% intervals over the 100 articles (2,000 resamples) [20], [21]. We call a difference clear when its interval excludes zero. For a human check, 48 summaries (8 articles × levels 0, 40, 80 × two prompts) were tagged blind to condition for entity swaps, number drift, date shifts, unsupported additions and language drift."),

  H1("V", "Results"),
  H2("A", "Effect of code-mixing"),
  P("Table I gives the main results for Llama 3.2 1B. Fact recall does not fall with mixing: with the plain prompt it stays between 0.306 and 0.327 at every level (level 80 minus level 0: 0.000, interval −0.025 to +0.024). Wrong facts, however, increase. Precision falls from 0.969 to 0.923 (−0.046, interval −0.081 to −0.015) and unsupported facts per summary more than double, from 0.16 to 0.38 (+0.22, 0.07 to 0.39). Ignoring English words copied from the input changes level-80 precision only from 0.923 to 0.930, so the drop is not an artifact of the entity tagger. The model writes in Hindi at every level: at most 1.3% of summary words contain Latin letters."),
  ...table("Table I", "Llama 3.2 1B: effect of mixing level and prompt",
    ["Level", "Prompt", "Recall", "Prec.", "F1", "Unsup.", "Compr.", "Time (s)"],
    [["0", "plain", ".306", ".969", ".446", ".16", ".196", "6.4"], ["0", "checklist", ".344", ".976", ".482", ".12", ".194", "6.5"],
     ["20", "plain", ".313", ".943", ".443", ".28", ".201", "6.5"], ["20", "checklist", ".336", ".960", ".474", ".24", ".193", "6.5"],
     ["40", "plain", ".327", ".957", ".461", ".24", ".197", "6.4"], ["40", "checklist", ".333", ".952", ".471", ".31", ".198", "6.6"],
     ["60", "plain", ".314", ".937", ".452", ".31", ".208", "6.7"], ["60", "checklist", ".368", ".960", ".510", ".24", ".202", "6.5"],
     ["80", "plain", ".307", ".923", ".446", ".38", ".206", "6.5"], ["80", "checklist", ".363", ".942", ".499", ".34", ".199", "6.5"]],
    [560, 820, 600, 600, 560, 600, 640, 660],
    "Unsup. = unsupported facts per summary; Compr. = compression ratio; Time = seconds per summary on a T4. 100 articles per row."),
  ...figure("results/chart.png", 1800, 760, "Fig. 1. Fact recall and precision of Llama 3.2 1B summaries against mixing level, for the plain and checklist prompts."),
  P("The fact-checklist prompt raises recall at every level, by 0.006 to 0.056, with clear gains at levels 0 and 80, and most for numbers (number recall 0.490 vs 0.327 at level 80). It does not lengthen the summaries (54–58 words for both prompts, compression about 0.20) and costs no extra time (about 6.5 s per summary)."),
  H2("B", "Model size and family"),
  P("Table II shows the effect of mixing across all five models. Every Llama model gains unsupported facts with mixing, with intervals clear of zero, even though the 8B model starts from far fewer (0.06 per summary). Llama 3B is no better than 1B. For both Gemma 2 models the rise is small and within noise, so Gemma is the least affected family, although the intervals overlap with Llama 8B’s. Larger models do not recall more facts with the plain prompt (0.28–0.33 everywhere); they follow the length instruction and write shorter summaries. The checklist prompt raises recall for every model and level, most for Gemma 2 9B (+0.11 to +0.14)."),
  ...table("Table II", "Effect of mixing across models (plain prompt)",
    ["Model", "Unsup. L0→L80", "Δ (95% CI)", "Contr. L0→L80", "F1 plain / ckl.", "Time (s)"],
    [["Llama 1B", ".16→.38", "+.22 (.07, .39)", ".19→.41", ".451 / .484", "6.5"],
     ["Llama 3B", ".15→.31", "+.16 (.04, .29)", ".18→.16", ".420 / .471", "7.2"],
     ["Llama 8B", ".06→.18", "+.12 (.03, .24)", ".13→.07", ".431 / .498", "9.3"],
     ["Gemma 2B", ".16→.24", "+.08 (−.04, .23)", ".09→.17", ".426 / .492", "9.1"],
     ["Gemma 9B", ".35→.37", "+.02 (−.09, .13)", ".10→.13", ".431 / .566", "19.2"]],
    [800, 840, 1060, 800, 900, 640],
    "Contr. = share of summaries with an NLI-contradicted sentence. F1 is averaged over levels 0/40/80. Gemma runs used float32 compute, so their times are not directly comparable with Llama’s."),
  H2("C", "English fact words"),
  P("When the fact words themselves are switched to English (targeted) rather than the same number of other words (random), the plain prompt loses recall for every model (Llama 1B −0.044, 3B −0.038, 8B −0.031, Gemma 2B −0.028, all clear; Gemma 9B −0.015, within noise) and precision for four of the five (Llama 1B −0.053, 3B −0.082). Most of the precision loss is respelling rather than invention: the models write the English name back in Devanagari in a spelling that differs from the article’s (24 of 31 unsupported entities for Llama 1B are close spellings of a checklist name). The checklist prompt, which shows the article’s own spellings, closes most of this gap (Llama 1B −0.013 recall and −0.006 precision, both within noise)."),
  H2("D", "Meaning preservation"),
  P("On the 48 hand-tagged summaries, the NLI check agrees moderately with the human tags: summaries with a factual error have lower NLI support (0.58 vs 0.86; AUC 0.74) and are flagged as contradicted more often (52% vs 12%). It catches about half of the entity and role swaps that fact matching misses. Applied to 3,000 summaries, it shows that mixing doubles the share of Llama 1B summaries with a contradicted sentence (0.19 to 0.41, +0.22, 0.12 to 0.32), while no larger model shows a clear rise with the plain prompt (Table II). The checklist prompt does not change NLI support for any model: it makes models mention more facts, but not relate them more correctly."),
  H2("E", "Content preservation"),
  P("Without mixing, every model beats the Lead-3 baseline on ROUGE-1 against the reference summary (0.27–0.30 vs 0.255). Mixing lowers ROUGE-1 clearly for Llama 1B (−0.032) and 3B (−0.026) and not clearly for the other models; BERTScore shows the same order. The checklist prompt does not change ROUGE, so its recall gains come from keeping specific names and numbers rather than from writing closer to the human summary. Absolute ROUGE is low because the reference is a single sentence and our summaries are zero-shot and up to three sentences long."),
  H2("F", "Prompting strategies"),
  P("Table III and Fig. 2 compare the four prompts. For Gemma 2 9B, explicit rules help: the instructed prompt raises recall over the checklist prompt at levels 0 and 40 (+0.035, clear) and gives the best F1 in the study (0.588–0.594). At level 80 it cuts unsupported facts from 0.22 to 0.08 per summary (−0.14, interval −0.24 to −0.05), and the effect of mixing on unsupported facts, still present with the checklist (+0.11, 0.03 to 0.19), disappears (+0.01, −0.06 to 0.07). It costs no extra time. For Llama 1B the rules move every measure in the right direction but no difference is clear, mixing still adds 0.22 unsupported facts per summary, and 95–97% of its outputs run into the token limit."),
  ...table("Table III", "Four prompting strategies at mixing level 80",
    ["Model", "Prompt", "Recall", "Prec.", "F1", "Unsup. L0/L80", "Contr.", "Time (s)"],
    [["Llama 1B", "plain", ".307", ".923", ".446", ".16 / .38", ".41", "6.5"], ["", "checklist", ".363", ".942", ".499", ".12 / .34", ".42", "6.5"],
     ["", "instructed", ".363", ".951", ".505", ".07 / .29", ".37", "7.1"], ["", "verified", ".248", ".741*", ".352", ".60 / .79*", ".26", "13.1"],
     ["Gemma 9B", "plain", ".304", ".919", ".439", ".35 / .37", ".13", "18.1"], ["", "checklist", ".440", ".964", ".577", ".11 / .22", ".18", "20.3"],
     ["", "instructed", ".452", ".985", ".594", ".07 / .08", ".19", "19.6"], ["", "verified", ".393", ".976", ".539", ".10 / .13", ".20", "36.8"]],
    [760, 780, 560, 560, 520, 900, 480, 480],
    "*Inflated by list numbering in the model’s output (“1.”, “2.” counted as numbers); with list markers removed, precision is .931 and unsupported facts .12 / .30. Verified time includes both passes."),
  ...figure("paper/fig_prompting.png", 1050, 1320, "Fig. 2. Fact recall, unsupported facts per summary and share of summaries with an NLI-contradicted sentence, against mixing level, for four prompts (left: Llama 3.2 1B; right: Gemma 2 9B). Llama 1B’s verified line for unsupported facts includes the list-numbering artifact (Table III)."),
  P("Self-verification by the same model does not help. For both models the check pass shortens the summaries (Gemma 49 to 44 words) and removes facts rather than errors: recall falls clearly at every level (Gemma −0.05 to −0.065, Llama −0.12 to −0.15 relative to instructed) while unsupported facts do not fall, at twice the cost. Llama 1B often breaks the format in this pass, answering as a numbered list or looping, which inflates its unsupported numbers (Table III). Its one gain is fewer summaries with a contradicted sentence (0.26 vs 0.37 at level 80), partly because shorter summaries have fewer sentences to contradict."),
  H2("G", "Error analysis"),
  P("In the 48 hand-tagged Llama 1B summaries, errors grow with mixing: 5 of 16 summaries have an error at level 0, 10 of 16 at level 40 and 14 of 16 at level 80. The most common factual error is an entity swap (15 tags), and many are _role_ swaps that keep the right names: “the BJP tried to buy Jogi’s legislators” where the article says Jogi tried to buy the BJP’s. String-based precision cannot detect these, which is why we add NLI. Language drift (English words kept or garbled in the summary) appears only with mixed input. No date shifts were found."),

  H1("VI", "Discussion and Limitations"),
  P("Our results suggest that code-mixing harms faithfulness more than coverage: models keep the facts but add or distort others, and the harm is largest for small models and for English fact words. A factuality-aware instruction is the most effective remedy we tested, but only for a model strong enough to follow it; self-verification by the same model is not a cheap fix."),
  P("The study has limitations. The mixed text is machine-made by dictionary lookup, not written by people; it never mixes verbs, reaches at most 18.6% English words, and sometimes picks a wrong word sense. Recall and precision rely on string matching against an automatic checklist, so spelling variants count as errors and paraphrased facts as missed. The NLI check agrees only moderately with human judgment (AUC 0.69–0.74 on 48 summaries). The human sample is small, and the final tags were audited by an AI assistant and reviewed by the author. The two newer prompts were tested on two models and three levels, and many comparisons are made without multiple-comparison correction. ROUGE uses our own Hindi tokenization and is not comparable with published XL-Sum scores."),

  H1("VII", "Conclusion"),
  P("We built controlled code-mixed versions of 100 Hindi news articles and measured how five open LLMs summarize them under four prompting strategies. Code-mixing does not make models lose facts, but it makes the smaller and Llama models add unsupported ones and distort meaning, especially when the fact words themselves are in English. Giving the model the article’s facts raises recall for every model, and explicit factuality rules let the strongest model (Gemma 2 9B) reach the best F1 while neutralizing the effect of mixing on unsupported facts. Future work should test naturally written Hinglish, a separate and stronger verifier, and checks that capture role errors directly."),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 202 }, spacing: { line: 228, after: 0 }, children: [
    new TextRun({ text: "Code, data and all outputs are available at ", font: FONT, size: 20 }),
    new ExternalHyperlink({ link: "https://github.com/Swapnil-Pramanik/FactIndic", children: [
      new TextRun({ text: "github.com/Swapnil-Pramanik/FactIndic", font: FONT, size: 20, style: "Hyperlink" })] }),
    new TextRun({ text: ".", font: FONT, size: 20 }),
  ] }),

  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 200, after: 100 }, keepNext: true,
    children: [new TextRun({ text: "References", font: FONT, size: 20, smallCaps: true })] }),
];

const REFS = [
  "J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, “On faithfulness and factuality in abstractive summarization,” in Proc. ACL, 2020, pp. 1906–1919.",
  "W. Kryściński, B. McCann, C. Xiong, and R. Socher, “Evaluating the factual consistency of abstractive text summarization,” in Proc. EMNLP, 2020, pp. 9332–9346.",
  "S. Khanuja, S. Dandapat, A. Srinivasan, S. Sitaram, and M. Choudhury, “GLUECoS: An evaluation benchmark for code-switched NLP,” in Proc. ACL, 2020, pp. 3575–3585.",
  "G. Aguilar, S. Kar, and T. Solorio, “LinCE: A centralized benchmark for linguistic code-switching evaluation,” in Proc. LREC, 2020, pp. 1803–1813.",
  "P. Laban, T. Schnabel, P. N. Bennett, and M. A. Hearst, “SummaC: Re-visiting NLI-based models for inconsistency detection in summarization,” Trans. Assoc. Comput. Linguistics, vol. 10, pp. 163–177, 2022.",
  "T. Hasan et al., “XL-Sum: Large-scale multilingual abstractive summarization for 44 languages,” in Findings of ACL-IJCNLP, 2021, pp. 4693–4703.",
  "A. Kamath, K. Singla, R. Paul, R. Joshi, U. Vaidya, S. S. Chauhan, and N. Wartikar, “Benchmarking Hindi LLMs: A new suite of datasets and a comparative analysis,” in Proc. 1st Workshop on Benchmarks, Harmonization, Annotation, and Standardization for Human-Centric AI in Indian Languages (BHASHA), 2025.",
  "A. Mhaske et al., “Naamapadam: A large-scale named entity annotated data for Indic languages,” in Proc. ACL, 2023, pp. 10441–10456.",
  "P. Qi, Y. Zhang, Y. Zhang, J. Bolton, and C. D. Manning, “Stanza: A Python natural language processing toolkit for many human languages,” in Proc. ACL: System Demonstrations, 2020, pp. 101–108.",
  "A. Conneau, G. Lample, M. Ranzato, L. Denoyer, and H. Jégou, “Word translation without parallel data,” in Proc. ICLR, 2018.",
  "A. Grattafiori et al., “The Llama 3 herd of models,” arXiv:2407.21783, 2024.",
  "Gemma Team, “Gemma 2: Improving open language models at a practical size,” arXiv:2408.00118, 2024.",
  "T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient finetuning of quantized LLMs,” in Proc. NeurIPS, 2023.",
  "C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out (ACL Workshop), 2004, pp. 74–81.",
  "T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating text generation with BERT,” in Proc. ICLR, 2020.",
  "A. Conneau et al., “Unsupervised cross-lingual representation learning at scale,” in Proc. ACL, 2020, pp. 8440–8451.",
  "M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers, “Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI,” Political Analysis, vol. 32, no. 1, pp. 84–100, 2024.",
  "P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing,” in Proc. ICLR, 2023.",
  "A. Conneau et al., “XNLI: Evaluating cross-lingual sentence representations,” in Proc. EMNLP, 2018, pp. 2475–2485.",
  "B. Efron and R. J. Tibshirani, An Introduction to the Bootstrap. New York, NY, USA: Chapman & Hall, 1993.",
  "P. Koehn, “Statistical significance tests for machine translation evaluation,” in Proc. EMNLP, 2004, pp. 388–395.",
];
const refParas = REFS.map((r, i) => new Paragraph({
  alignment: AlignmentType.JUSTIFIED, indent: { left: 360, hanging: 360 }, spacing: { after: 30, line: 200 },
  children: [new TextRun({ text: `[${i + 1}]\t${r}`, font: FONT, size: 16 })],
  tabStops: [{ type: "left", position: 360 }],
}));

const doc = new Document({
  creator: "Swapnil Pramanik",
  title: "Do Code-Mixed Inputs Make LLM Summaries Less Faithful?",
  styles: { default: { document: { run: { font: FONT, size: 20 } } },
    characterStyles: [{ id: "Hyperlink", name: "Hyperlink", run: { color: "000000" } }] },
  sections: [
    { properties: { page: PAGE }, children: front },
    { properties: { page: PAGE, type: SectionType.CONTINUOUS, column: { count: 2, space: 360, equalWidth: true } },
      children: [...body, ...refParas] },
    // An empty continuous section at the end makes Word balance the two columns on the last page.
    { properties: { page: PAGE, type: SectionType.CONTINUOUS }, children: [new Paragraph({ children: [] })] },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "FactIndic_IEEE.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote " + out);
});
