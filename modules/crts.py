import numpy as np

def compute_crts(
    attestations,
    conflict_summary,
    surfaced_risks_count=0,
    nice_matches=None
):
    """
    Computes the Clinical RAG Transparency Score (CRTS) according to
    the formal definitions described in the manuscript.

    Components:
    - Source Fidelity (SF)
    - Conflict Reporting Rate (CRR)
    - Audit Responsiveness (AR = 1 / L)
    - Guideline Alignment (GA)

    Parameters
    ----------
    attestations : dict
        Mapping of generated claims to attestation metadata.
        A claim is considered grounded if it has non-empty evidence.

    conflict_summary : dict
        Output from detect_conflicts(), containing detected risk signals.

    surfaced_risks_count : int
        Number of risk or safety signals explicitly surfaced in the synthesis.

    nice_matches : list or None
        List of NICE guideline matches. Empty or None implies no alignment.

    Returns
    -------
    dict
        CRTS components and raw audit latency.
    """

    # ------------------------------------------------------------
    # 1. Source Fidelity (SF)
    # ------------------------------------------------------------
    n_claims = len(attestations)

    if n_claims > 0:
        sf = sum(
            1 for v in attestations.values() if v
        ) / n_claims
    else:
        sf = 0.0

    # ------------------------------------------------------------
    # 2. Conflict Reporting Rate (CRR)
    # ------------------------------------------------------------
    detected_conflicts = conflict_summary.get("risk", 0)

    if detected_conflicts > 0:
        crr = min(1.0, surfaced_risks_count / detected_conflicts)
    else:
        # No conflicts detected implies perfect reporting
        crr = 1.0

    # ------------------------------------------------------------
    # 3. Audit Responsiveness (AR = 1 / L)
    # ------------------------------------------------------------
    # Granular attestation (sentence-level) assumed faster audit
    has_granular_attestation = any(
        isinstance(v, dict) and "source_text" in v
        for v in attestations.values()
    )

    # Latency proxy (seconds)
    L = 2 if has_granular_attestation else 5
    ar = 1 / L

    # ------------------------------------------------------------
    # 4. Guideline Alignment (GA)
    # ------------------------------------------------------------
    ga = 1.0 if nice_matches else 0.0

    # ------------------------------------------------------------
    # Final CRTS component dictionary
    # ------------------------------------------------------------
    return {
        "sf": round(sf, 2),
        "crr": round(crr, 2),
        "ar": round(ar, 2),
        "ga": round(ga, 2),
        "L": L
    }
