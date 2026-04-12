# Clinical RAG Active Verifiability Framework – Model Card

## 1. Model/System Overview

**System name:** Clinical RAG Active Verifiability Framework (AV-RAG)  
**Version:** v0.1 (hybrid: heuristics + Agents A/B + CRTS)  
**Primary authors / maintainers:** Anthony Onoja, School of Health Sciences, University of Surrey, UK (a.onoja@surrey.ac.uk)  
**Source repository:** [...]  

This system is a **clinical research decision-support tool**, not a diagnostic or prescribing system. It implements the **Active Verifiability (AV)** framework described in:

> Onoja et al., "Navigating the Black Box of Clinical RAG: A Framework for Active Verifiability and Patient Safety", [...]

The architecture emphasises:

- **Contradiction-aware retrieval**
- **Claim-level attestation**
- **Dynamic guideline anchoring**
- **Clinical RAG Transparency Score (CRTS)**
- **Clinician Validation Loop (Human-in-the-Loop)**

The system uses large language models (LLMs) as **transparent agents**, not as a single opaque judge, while keeping the clinician actively engaged through real-time feedback mechanisms.

---

## 2. Intended Use

### 2.1 Intended Users

- Clinical researchers  
- Clinician–scientists  
- Health policy analysts  
- Clinical AI safety and governance teams  

### 2.2 Intended Use Cases

- Exploring the biomedical literature (e.g. PubMed) for a clinical question.
- Inspecting **evidentiary dissent** (benefit vs risk signals).
- Checking whether generated claims are:
  - Grounded in specific abstracts (attestation).
  - Aligned with selected clinical guidelines (e.g. NICE, local protocols).
- Monitoring **CRTS** as a summary of transparency and auditability, not accuracy.

### 2.3 Out of Scope

- Making treatment decisions for individual patients.  
- Providing final therapeutic recommendations.  
- Replacing formal guideline appraisal, systematic reviews, or clinical reasoning.

The UI explicitly states:  
> "Research decision-support tool. Not for clinical diagnosis or treatment."

---

## 3. System Architecture

### 3.1 High-Level Pipeline

1. **Retrieval & Prioritisation**
   - PubMed search via NCBI (email: `NCBI_EMAIL` in `config/settings.py`).
   - Up to `MAX_PUBMED_RESULTS = 1000` results.
   - `prioritise_documents` ranks abstracts using relevance and recency.

2. **Conflict Detection**
   - `detect_conflicts` labels documents as **supportive** or **risk-signalling** for the query.
   - Produces `conflict_summary = {"supportive": n, "risk": n, "doc_tags": [...]}`.

3. **Agent A – Evidence Analyser (LLM-based)**
   - `agents.agent_evidence_analyser.EvidenceAnalyserAgent` (Gemini).
   - For each abstract, extracts **atomic claims** with:
     - `stance ∈ {supportive, risk, neutral, inconclusive}`
     - `evidence_type ∈ {direct_evidence, interpretive}`
     - `outcome_type ∈ {efficacy, safety, other, unclear}`
   - Outputs: `evidence_claims` (structured JSON).

4. **Transparent Synthesis & Attestation**
   - `modules.attestation.generate_with_attestation` (Gemini + deterministic fallback).
   - Produces:
     - `synthesis`: concise narrative.
     - `attestations: {claim_text -> {pmid, source_text, document_title}}`.

5. **Guideline Anchoring**
   - `modules.guideline_retrieval`:
     - `scrape_guideline_url` (web, e.g. NICE).
     - `process_uploaded_pdf` (local protocols).
   - `align_claims_to_guidelines` + `compute_ga_metrics`:
     - TF-IDF + cosine similarity from claims to guideline chunks.
     - GA = proportion of attested claims with at least one above-threshold match.

6. **Agent B – Guideline Comparator (LLM-based)**
   - `agents.agent_guideline_comparator.GuidelineComparatorAgent` (Gemini).
   - For each claim, uses top-k guideline chunks to classify:
     - `alignment_label ∈ {aligned, partially_aligned, contradicted, not_addressed}`
     - plus a textual rationale.
   - This does **not** currently feed back into GA; it is an additional transparency view.

7. **CRTS Computation**
   - `modules.crts.compute_crts`:
     - **SF (Source Fidelity):** fraction of generated claims with valid attestations.
     - **CRR (Conflict Reporting Rate):** surfaced risk/conflict signals / detected risk signals.
     - **AR (Audit Responsiveness):** inverse audit latency (seconds per claim, capped).
     - **GA (Guideline Alignment):** from TF-IDF GA metric.
     - **CRTS composite:**  
       $$\text{CRTS} = \alpha \cdot SF + \beta \cdot CRR + \gamma \cdot AR^* + \delta \cdot GA$$
     - Default weights (normalised): 
       - $$\alpha = 0.30, \beta = 0.30, \gamma = 0.20, \delta = 0.20$$ (overridable in UI).

8. **Clinician Validation Loop (HITL)**
   - An interactive UI mechanism where the clinician provides active feedback on the AI’s synthesis and reasoning.
   - Logs the validation output (e.g., rating, remarks) and maps it back to the original query and generated claims via `human_feedback_log.jsonl`.
   - Ensures continuous evaluation by domain experts, keeping humans firmly "in the loop".

9. **Visualisation & Logging**
   - Evidence balance bar chart (supportive vs risk-signalling studies).
   - CRTS radar chart.
   - Evidence similarity network (PyVis).
   - Audit logs: `modules.logging.log_crts_both` records CRTS and metadata to CSV/JSONL, now accompanied by human validation logs.

---

## 4. Data and LLMs

### 4.1 Data Sources

- **PubMed abstracts**, retrieved via NCBI e-utilities (NCBI email and rate limits respected).
- **Clinical guidelines**:
  - NICE guidelines (e.g. NG14, NG141) via public web scraping.
  - Local institutional PDFs via upload (parsed with `pdfplumber` or similar).

No patient-identifiable data are used. All data are **secondary, published literature or guidelines**.

### 4.2 LLMs

Current default LLM: **Google Gemini 2.5 (flash-lite)**

- Used in:
  - `generate_with_attestation` (LLM path).
  - `EvidenceAnalyserAgent` (Agent A).
  - `GuidelineComparatorAgent` (Agent B).

The system is designed to be **LLM-pluggable**; alternative providers can be integrated (e.g. local llama) while preserving CRTS.

---

## 5. Evaluation and Benchmarks

### 5.1 Evaluation Setup

- Evaluation script: `evaluation/run_benchmark.py`
- Dataset: `evaluation/cases.jsonl`
  - Each case consists of:
    - A clinical query.
    - A set of **gold-labelled claims** with:
      - `stance ∈ {supportive, risk, neutral, inconclusive}`
      - `is_direct_evidence ∈ {true, false}`
      - `guideline_alignment ∈ {aligned, contradicted, not_addressed}`

- For each case, the script:
  1. Runs the full pipeline (retrieval → Agents → CRTS).
  2. Evaluates:
     - Agent A **stance accuracy** vs gold.
     - Agent B **guideline alignment accuracy** vs gold.
     - TF-IDF **GA score**.
     - CRTS components and composite.

### 5.2 Metrics (to be filled after running the benchmark)

Fill these once you have real numbers from `evaluation/results.jsonl`:

#### 5.2.1 Agent A – Stance Classification

- Number of cases: `N_cases = [...]`
- Total gold-labelled claims used: `N_gold_claims = [...]`
- Agent A stance accuracy: `stance_accuracy_agentA = [...]`  
  (You can also report precision/recall/F1 per class.)

#### 5.2.2 Agent B – Guideline Alignment Classification

- Total gold-labelled aligned/contradicted/not_addressed claims: `N_gold_guideline = [...]`
- Agent B alignment accuracy: `guideline_alignment_accuracy_agentB = [...]`

Optionally, report:
- Accuracy by class (aligned / contradicted / not_addressed).
- Confusion matrix.

#### 5.2.3 TF-IDF GA vs Agent B

- Mean GA score (TF-IDF): `mean_ga_tfidf = [...]`
- Mean GA by Agent B (`aligned` / `partially_aligned` / others): `[...]`
- Observations on where Agent B and TF-IDF disagree.

#### 5.2.4 CRTS Distributions

Across all cases:

- $$\overline{SF} = [...]$$  
- $$\overline{CRR} = [...]$$  
- $$\overline{AR^*} = [...]$$  
- $$\overline{GA} = [...]$$  
- $$\overline{CRTS} = [...]$$  

If you collect human ratings (e.g. `gold_trust_score ∈ [0,1]` per case), also report:

- Correlation between CRTS and human trust score: `corr(CRTS, trust) = [...]`.

### 5.3 Qualitative Findings

Summarise:

- Cases where **high SF but low CRR** uncovered safety-relevant dissent that would be missed by a naive RAG.  
- Scenarios where Agent B detected **guideline contradiction** despite TF-IDF alignment being high (phantom policy alignment).  
- Limitations observed:
  - Over-sensitivity to paraphrasing.
  - Missed minority risk signals in certain domains.

---

## 6. Limitations

### 6.1 Methodological Limitations

- Evaluation is **methods-oriented** (transparency, dissent visibility, guideline alignment), not a clinical outcome trial.
- Gold labels rely on expert annotation of:
  - Stance of claims.
  - Guideline alignment.
  - Direct vs interpretive evidence.
- PubMed retrieval:
  - May miss relevant newer or non-indexed studies.
  - Is susceptible to keyword choice in the query.

### 6.2 LLM Limitations

- Gemini outputs are **non-deterministic**; stance and alignment labels can vary slightly across runs.
- The LLM may:
  - Misinterpret ambiguous sentences.
  - Under- or over-call contradiction vs partial alignment.
- Agent outputs are **prompt-sensitive**; future prompt revisions may change behaviour.

### 6.3 Clinical Scope

- The system's initial focus included **oncology / immunotherapy examples** (e.g. PD-1 inhibitors and melanoma), but has been expanded following domain expert requirements to encompass **niche vestibular physiotherapy and neurology** (e.g., Persistent Postural-Perceptual Dizziness / PPPD).
- The framework natively supports specialised local protocols such as NHS Lanarkshire physiotherapy, Physio-pedia clinical guidance, and formal NICE/WHO guidelines.
- Generalisation to other unverified specialties requires ongoing calibration via our new Clinician Validation Loop.
- Guideline anchoring:
  - Heavily dependent on quality and recency of chosen guideline source (e.g. NICE vs local protcols).

---

## 7. Ethical and Safety Considerations

- The system emphasises **epistemic transparency** rather than providing a single “correct answer”.
- **CRTS** shifts focus from correctness alone to:
  - **Source fidelity**
  - **Conflict reporting**
  - **Audit responsiveness**
  - **Guideline alignment**
- High CRTS does **not** imply correct clinical decisions; it indicates that the system’s outputs are easier to check and contest.

Key safety measures:

- Clear UI disclaimers (“Not for clinical diagnosis or treatment”).
- Distinct labelling of **interpretive claims** (Agent A) vs direct evidence.
- Explicit surfacing of **risk-signalling evidence** and **guideline contradictions** (Agent B).
- Logging of runs to support auditability and regulatory review.

---

## 8. Versioning and Future Work

- This model card describes **version v0.1** of the AV-RAG framework.
- Planned work:
  - Larger, domain-diverse evaluation set.
  - Extended metrics for claim-level attestation fidelity.
  - Comparison of different LLM backends (Gemini vs local llama vs GPT) under the same CRTS framework.
  - Integration of Agent C (CRTS explainer) as a narrative adjunct, not a scorer.

---

If you like, I can next help you write a **small analysis notebook or script** that reads `evaluation/results.jsonl` and automatically computes the summary metrics to paste into the “Metrics” section of the model card.