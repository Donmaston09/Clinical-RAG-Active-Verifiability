def detect_conflicts(documents):
    """
    Detects evidence-level conflict by identifying supportive versus
    risk-signalling language across retrieved studies.
    """

    supportive_terms = [
        "improves", "improved", "benefit", "effective", "efficacy",
        "survival advantage", "response rate"
    ]

    risk_terms = [
        "toxicity", "adverse", "risk", "harm", "side effect",
        "complication", "safety concern"
    ]

    supportive_count = 0
    risk_count = 0

    for doc in documents:
        text = doc.get("abstract", "").lower()

        if any(term in text for term in supportive_terms):
            supportive_count += 1

        if any(term in text for term in risk_terms):
            risk_count += 1

    return {
        "detected": supportive_count > 0 and risk_count > 0,
        "supportive": supportive_count,
        "risk": risk_count
    }
