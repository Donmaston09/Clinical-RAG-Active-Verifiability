
"""
Conflict detection utilities
- Detects supportive vs risk-signalling language across retrieved studies.
- Emits both corpus-level counts and per-document tags (for downstream UIs/metrics).
This module is intentionally keyword-based (deterministic, auditable) to avoid
opaque model decisions during safety-critical review.
"""
from typing import List, Dict
import re

SUPPORTIVE_TERMS = [
    r"improv(e|ed|ement)", r"benefit(s|ed)?", r"effective|efficacy",
    r"survival advantage", r"response rate", r"superior(ity)?"
]
RISK_TERMS = [
    r"toxicit(y|ies)", r"adverse( event|s)?", r"risk(s)?", r"harm(s|ful)?",
    r"side effect(s)?", r"complication(s)?", r"safety concern(s)?", r"contraindicat(ed|ion)"
]
SUPPORTIVE_RE = re.compile(r"|".join(SUPPORTIVE_TERMS), re.I)
RISK_RE = re.compile(r"|".join(RISK_TERMS), re.I)


def tag_document(doc: Dict) -> Dict:
    """Returns a tag summary for a single document.
    Expects keys: 'pmid', 'title', 'abstract', 'publication_type' (optional list).
    """
    text = f"{doc.get('title','')} {doc.get('abstract','')}"
    sup_hits = SUPPORTIVE_RE.findall(text or "")
    risk_hits = RISK_RE.findall(text or "")
    return {
        'pmid': doc.get('pmid'),
        'supportive': bool(sup_hits),
        'risk': bool(risk_hits),
        'support_terms': list(set([h if isinstance(h,str) else ''.join(h) for h in sup_hits])),
        'risk_terms': list(set([h if isinstance(h,str) else ''.join(h) for h in risk_hits]))
    }


def detect_conflicts(documents: List[Dict]) -> Dict:
    """Detects evidence-level conflict by identifying supportive versus risk-signalling
    language across retrieved studies. Returns aggregate counts and doc-level tags.
    """
    supportive_count = 0
    risk_count = 0
    doc_tags = []
    for doc in documents:
        tags = tag_document(doc)
        doc_tags.append(tags)
        supportive_count += 1 if tags['supportive'] else 0
        risk_count += 1 if tags['risk'] else 0
    return {
        'detected': supportive_count > 0 and risk_count > 0,
        'supportive': supportive_count,
        'risk': risk_count,
        'doc_tags': doc_tags,
    }
