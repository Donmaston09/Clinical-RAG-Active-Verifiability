import json
import os
from typing import Dict, Any, List
from datetime import datetime

from modules.pubmed_retrieval import search_pubmed, fetch_abstracts
from modules.evidence_scoring import prioritise_documents
from modules.conflict_detection import detect_conflicts
from modules.attestation import generate_with_attestation
from modules.guideline_retrieval import (
    align_claims_to_guidelines,
    process_uploaded_pdf,
    scrape_guideline_url,
)
from modules.guideline_checker import compute_ga_metrics
from modules.crts import compute_crts

from agents.agent_evidence_analyser import EvidenceAnalyserAgent
from agents.agent_guideline_comparator import GuidelineComparatorAgent


EVAL_CASES_PATH = "evaluation/cases.jsonl"
EVAL_OUTPUT_PATH = "evaluation/results.jsonl"

# Configure a default guideline source for benchmarking
DEFAULT_GUIDELINE_URL = "https://www.nice.org.uk/guidance/ng14"


def load_eval_cases(path: str) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cases.append(obj)
            except json.JSONDecodeError as e:
                print(f"[WARNING] Skipping invalid JSON on line {i}: {e}")
                print(f"  Line content: {line}")
                continue
    return cases

def run_single_case(
    case: Dict[str, Any],
    api_key: str = None,
    guideline_chunks: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the full pipeline on a single evaluation case and compute metrics.
    Returns a dict suitable for writing to JSONL.
    """
    query = case["query"]

    # 1) Retrieval + prioritisation
    pmids = search_pubmed(query)
    raw_docs = fetch_abstracts(pmids)
    documents = prioritise_documents(raw_docs, query)

    # 2) Conflict detection
    conflict_summary = detect_conflicts(documents)

    # 3) Agent A: evidence analyser
    evidence_agent = EvidenceAnalyserAgent(api_key=api_key or None)
    evidence_claims = evidence_agent.analyse_docs(documents)

    # 4) Synthesis + attestation
    synthesis, attestations = generate_with_attestation(query, documents, api_key=api_key)

    # 5) Guideline chunks
    if guideline_chunks is None:
        guideline_chunks = scrape_guideline_url(DEFAULT_GUIDELINE_URL)

    # 6) Guideline alignment (existing TF-IDF GA)
    attested_claims = list(attestations.keys())
    guideline_alignment = align_claims_to_guidelines(attested_claims, guideline_chunks)
    ga_score, ga_matched, ga_total = compute_ga_metrics(guideline_alignment)

    # 7) Agent B: guideline comparator
    guideline_agent = GuidelineComparatorAgent(api_key=api_key or None)
    guideline_assessments = guideline_agent.assess_claims(
        attested_claims, guideline_chunks, top_k=3
    )

    # 8) CRTS (SF, CRR, AR, GA, composite)
    detected_D = int(conflict_summary.get("risk", 0))

    # A simple surfaced risk estimate for the benchmark; same helper as app, or reuse from app module
    from app import estimate_surfaced_risks_from_claims  # if app.py is import-safe

    surfaced_S = estimate_surfaced_risks_from_claims(evidence_claims, synthesis, detected_D)

    crts = compute_crts(
        attestations=attestations,
        conflict_summary=conflict_summary,
        guideline_alignment=guideline_alignment,
        surfaced_risks_count=surfaced_S,
        k_seconds=5.0,
        weights={"alpha": 0.30, "beta": 0.30, "gamma": 0.20, "delta": 0.20},
    )

    # 9) Compare to gold labels

    gold_claims = case.get("gold_claims", [])
    def normalise(s: str) -> str:
        return " ".join((s or "").lower().split())

    stance_tp = 0
    stance_total = 0

    for g in gold_claims:
        gold_text_norm = normalise(g["claim_text"])
        gold_stance = g["stance"]

        # Find the first evidence_claim with high overlap
        best_match = None
        best_overlap = 0.0

        for ec in evidence_claims:
            ec_text_norm = normalise(ec["claim_text"])
            # simple overlap ratio: shared words / gold words
            gold_words = set(gold_text_norm.split())
            ec_words = set(ec_text_norm.split())
            if not gold_words:
                continue
            overlap = len(gold_words & ec_words) / len(gold_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = ec

        # Require at least 0.6 word overlap to consider it a match
        if best_match and best_overlap >= 0.6:
            stance_total += 1
            if best_match["stance"] == gold_stance:
                stance_tp += 1

    stance_acc = stance_tp / stance_total if stance_total else None
    # Evaluate guideline alignment for Agent B
    # Evaluate guideline alignment for Agent B (fuzzy match on claim text)
    ga_tp = 0
    ga_total_gold = 0

    def normalise(s: str) -> str:
        return " ".join((s or "").lower().split())

    for g in gold_claims:
        gold_label = g.get("guideline_alignment")
        if not gold_label:
            continue
        gold_text_norm = normalise(g["claim_text"])
        ga_total_gold += 1

        best_ass = None
        best_overlap = 0.0
        for a in guideline_assessments:
            a_text_norm = normalise(a["claim_text"])
            gold_words = set(gold_text_norm.split())
            a_words = set(a_text_norm.split())
            if not gold_words:
                continue
            overlap = len(gold_words & a_words) / len(gold_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_ass = a

        if best_ass and best_overlap >= 0.6:
            if best_ass["alignment_label"] == gold_label:
                ga_tp += 1

    ga_acc_agentB = ga_tp / ga_total_gold if ga_total_gold else None
    # Bundle result
    result = {
        "case_id": case["case_id"],
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "crts": crts,
        "conflict_summary": conflict_summary,
        "ga_score_tfidf": ga_score,
        "ga_matched_tfidf": ga_matched,
        "ga_total_tfidf": ga_total,
        "stance_accuracy_agentA": stance_acc,
        "guideline_alignment_accuracy_agentB": ga_acc_agentB,
        "n_evidence_claims_agentA": len(evidence_claims),
        "n_guideline_assessments_agentB": len(guideline_assessments),
    }

    return result


def main():
    api_key = os.getenv("GEMINI_API_KEY")  # or pass explicitly
    cases = load_eval_cases(EVAL_CASES_PATH)

    # Pre-load guideline chunks once for efficiency
    guideline_chunks = scrape_guideline_url(DEFAULT_GUIDELINE_URL)

    os.makedirs(os.path.dirname(EVAL_OUTPUT_PATH), exist_ok=True)
    with open(EVAL_OUTPUT_PATH, "a", encoding="utf-8") as f_out:
        for case in cases:
            print(f"Running case: {case['case_id']}...")
            res = run_single_case(case, api_key=api_key, guideline_chunks=guideline_chunks)
            f_out.write(json.dumps(res) + "\n")

    print(f"Evaluation finished. Results written to {EVAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
