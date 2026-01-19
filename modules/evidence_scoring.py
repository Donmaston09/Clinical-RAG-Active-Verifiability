
"""
Evidence prioritisation utilities
- Promotes high-value evidence types (RCTs, SRs), while preserving safety literature
  (post-marketing, pharmacovigilance) to avoid 'phantom consensus'.
- Deterministic, transparent feature-based scoring.
"""
from typing import List, Dict
import re

SAFETY_TERMS = [
    'adverse', 'toxicity', 'contraindicat', 'mortality', 'risk', 'harm',
    'side effect', 'complication', 'black box', 'warning'
]
HIGH_VALUE_TYPES = {
    'Randomized Controlled Trial': 1.0,
    'Systematic Review': 1.0,
    'Meta-Analysis': 0.9,
}
SAFETY_STUDY_HINTS = [
    'pharmacovigilance', 'post-marketing', 'registry', 'surveillance', 'real-world'
]


def _term_hits(text: str, terms) -> int:
    return sum(1 for t in terms if re.search(rf"{re.escape(t)}\w*", text, re.I))


def score_document(doc: Dict, query_terms: List[str]) -> float:
    score = 0.0
    title = doc.get('title', '')
    abstract = doc.get('abstract', '')
    text = f"{title} {abstract}".lower()

    # Relevance to query terms (bag-of-words)
    for term in query_terms:
        if term and term.lower() in text:
            score += 0.2

    # Evidence type weighting
    pub_types = doc.get('publication_type', []) or []
    for p in pub_types:
        score += HIGH_VALUE_TYPES.get(p, 0.0)

    # Safety signals (keep dissent visible)
    score += 0.25 * _term_hits(text, SAFETY_TERMS)

    # Prefer safety-focused/real-world reports (post-marketing, registries)
    score += 0.3 * _term_hits(text, SAFETY_STUDY_HINTS)

    # Recency boost (simple, deterministic)
    try:
        year = int(doc.get('year', 0))
        if year >= 2022:
            score += 0.4
        elif year >= 2019:
            score += 0.25
    except Exception:
        pass

    return round(score, 3)


def prioritise_documents(documents: List[Dict], query: str) -> List[Dict]:
    query_terms = [t for t in re.split(r"\W+", query) if t]
    for d in documents:
        d['priority_score'] = score_document(d, query_terms)
    # Deterministic sort (score desc, then PMID asc to stabilise ties)
    documents.sort(key=lambda x: (-x.get('priority_score', 0.0), str(x.get('pmid',''))))
    return documents
