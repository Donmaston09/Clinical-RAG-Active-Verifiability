import re

SAFETY_TERMS = [
    "adverse",
    "toxicity",
    "contraindicated",
    "mortality",
    "risk",
    "harm",
    "side effect",
    "complication"
]

HIGH_VALUE_TYPES = [
    "Randomized Controlled Trial",
    "Systematic Review",
    "Meta-Analysis"
]

def score_document(doc, query_terms):
    score = 0.0
    text = f"{doc['title']} {doc['abstract']}".lower()

    # Relevance to query terms
    for term in query_terms:
        if term.lower() in text:
            score += 0.2

    # Reward higher-quality evidence types
    if any(t in doc.get("publication_type", []) for t in HIGH_VALUE_TYPES):
        score += 0.5

    # Reward explicit safety signals (prevents phantom consensus)
    for safety_term in SAFETY_TERMS:
        if re.search(rf"\b{safety_term}\b", text):
            score += 0.4

    # Prefer recent literature
    try:
        if int(doc.get("year", 0)) >= 2019:
            score += 0.3
    except Exception:
        pass

    return round(score, 3)

def prioritise_documents(documents, query):
    query_terms = query.split()

    for d in documents:
        d["priority_score"] = score_document(d, query_terms)

    documents.sort(key=lambda x: x["priority_score"], reverse=True)
    return documents
