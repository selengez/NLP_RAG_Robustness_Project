AI USAGE DISCLAIMER

Parts of this project have been developed with the assistance of OpenAI’s Claude Sonnet 4.6. The AI was used to fix code errors, structure methodological workflows. Figures and graphs developed with AI assistance have been carefully reviewed, edited, and validated by me. I take full responsibility for the final content and its accuracy, relevance, and academic integrity.

# P5. In RAG We Trust?

This project evaluates how **Retrieval-Augmented Generation (RAG)** systems behave when the retrieved evidence is intentionally contaminated with false or conflicting information. Rather than proposing a new RAG architecture, the study focuses on **quantifying robustness** by measuring how different prompt strategies affect factual reliability, source grounding, and resistance to poisoned context.

**Core Pipeline Sketch:** A RAG pipeline is constructed with an intentionally poisoned retrieval setting. The system is tested against multiple forms of false, misleading, or conflicting evidence, while the model is prompted under different response strategies. Results are analysed using factual accuracy, hallucination, grounding, poison acceptance, and conflict-detection measures.

**Expected Outcomes:** The project provides a quantitative assessment of how retrieval-augmented generation systems handle misinformation in retrieved context, and whether prompt design can improve robustness against poisoned evidence.

---

## Objective

The core objective of this project is to quantitatively assess how robust a RAG pipeline remains when its document corpus is deliberately poisoned. The analysis is designed to identify:

- performance degradation as poison rate increases,
- differences between prompt strategies under poisoned retrieval conditions,
- and cases where the model incorrectly accepts misleading evidence.
---

## Implementation in This Study

In this study, I analyze the robustness of a RAG pipeline using two benchmark-style datasets:

- **FEVER**, for fact verification
- **HotpotQA**, for multi-hop question answering

Both clean and poisoned documents are mapped into the same retrieval space. The system retrieves relevant evidence, performs a pre-generation conflict check, and then produces an answer using a lightweight generation model.

The experiment systematically varies:

- **4 poisoning strategies**
- **4 poison rates**: 0%, 25%, 50%, 75%
- **3 prompt modes**

This setup makes it possible to compare how prompt design changes the model’s behaviour when the retrieved context becomes less trustworthy.

---

## Methodology

1. **Controlled Data Poisoning:** Introduce intentional falsehoods, contradictions, or misleading phrasing into retrieved document sets.
2. **Multi-source Verification:** Measure whether the model can detect inconsistencies, reject misleading evidence, or respond cautiously.
3. **Prompt Design:** Test whether verification-oriented or uncertainty-aware prompts improve robustness.
4. **Evaluation:** Compare accuracy, hallucination, grounding, poison acceptance, and conflict detection across poisoning settings.

---

## Poisoning Strategies

| Strategy | Description |
|---|---|
| `false_evidence` | Injects explicitly false or misleading evidence into the corpus |
| `conflicting_evidence` | Modifies dates, numbers, or key facts to create contradiction |
| `confident_false_evidence` | Presents false claims in a confident and authoritative style |
| `mixed_evidence` | Combines a partly correct statement with a false conclusion |

---

## Prompt Strategies

| Prompt Mode | Description |
|---|---|
| `standard_prompt` | Produces a direct answer from retrieved evidence |
| `verification_prompt` | Encourages the model to check whether sources contradict each other |
| `uncertainty_prompt` | Encourages the model to respond cautiously when evidence is weak or conflicting |

