
from typing import Dict, Any, Optional

DEFAULT_WEIGHTS = {"alpha": 0.30, "beta": 0.30, "gamma": 0.20, "delta": 0.20}


def compute_crts(
    attestations: Dict[str, Any],
    conflict_summary: Dict[str, Any],
    guideline_alignment: Optional[Dict[str, Optional[Dict[str, Any]]]] = None,
    surfaced_risks_count: int = 0,
    k_seconds: float = 5.0,
    weights: Dict[str, float] = DEFAULT_WEIGHTS,
) -> Dict[str, Any]:
    """
    Computes the Clinical RAG Transparency Score (CRTS) components with
    corrected definitions:
      - SF: proportion of claims with non-empty attestation
      - CRR: S/D if D>0 else 1, bounded to [0,1]
      - AR*: min(1, k/L) with L derived from attestation granularity
      - GA: proportion of claims with a guideline match (>= threshold)

    Returns dict with components, raw latency L, and composite.
    """
    # --- Source Fidelity ---
    n_claims = len(attestations)
    sf = (sum(1 for v in attestations.values() if v) / n_claims) if n_claims else 0.0

    # --- Conflict Reporting Rate ---
    detected_conflicts = int(conflict_summary.get("risk", 0))
    if detected_conflicts > 0:
        crr = min(1.0, max(0.0, surfaced_risks_count / detected_conflicts))
    else:
        crr = 1.0

    # --- Audit Responsiveness --- (L in seconds; AR* normalized)
    has_granular_attestation = any(isinstance(v, dict) and "source_text" in v for v in attestations.values())
    L = 2.0 if has_granular_attestation else 5.0
    ar = min(1.0, float(k_seconds) / float(L))  # AR* in [0,1]

    # --- Guideline Alignment --- (proportion of claims matched)
    if guideline_alignment and n_claims:
        ga = sum(1 for v in guideline_alignment.values() if v is not None) / n_claims
    else:
        ga = 0.0

    alpha = float(weights.get("alpha", 0.30))
    beta = float(weights.get("beta", 0.30))
    gamma = float(weights.get("gamma", 0.20))
    delta = float(weights.get("delta", 0.20))
    # Normalize weights to sum to 1 for safety
    total_w = alpha + beta + gamma + delta
    if total_w <= 0:
        alpha, beta, gamma, delta = 0.30, 0.30, 0.20, 0.20
        total_w = 1.0
    alpha, beta, gamma, delta = (alpha/total_w, beta/total_w, gamma/total_w, delta/total_w)

    composite = alpha*sf + beta*crr + gamma*ar + delta*ga

    return {
        "sf": round(sf, 2),
        "crr": round(crr, 2),
        "ar": round(ar, 2),
        "ga": round(ga, 2),
        "L": L,
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta},
        "crts": round(composite, 2),
    }
