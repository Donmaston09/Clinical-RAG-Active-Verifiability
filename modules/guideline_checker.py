
"""
Guideline alignment utilities complementing guideline_retrieval.align_claims_to_guidelines.
- Computes GA (proportion of claims aligned) and returns provenance summary.
- Keeps a legacy shim check_nice_alignment for backward compatibility.
"""
from typing import Dict, Optional, Tuple, List
import time


def compute_ga_metrics(alignment: Dict[str, Optional[Dict]]) -> Tuple[float, int, int]:
    """Return (GA, matched, total) from an alignment mapping claim->match or None."""
    total = len(alignment) if alignment else 0
    matched = sum(1 for v in (alignment or {}).values() if v is not None)
    ga = (matched / total) if total else 0.0
    return ga, matched, total


def provenance_summary(alignment: Dict[str, Optional[Dict]]) -> Dict[str, List[str]]:
    """Summarise sources and versioning metadata (Last-Modified, PDF hashes)."""
    sources, last_mod, hashes = set(), set(), set()
    for v in (alignment or {}).values():
        if not v:
            continue
        s = v.get('source')
        if s:
            sources.add(s)
        lm = v.get('last_modified')
        if lm:
            last_mod.add(lm)
        h = v.get('hash')
        if h:
            hashes.add(h)
    return {
        'sources': sorted(sources),
        'last_modified': sorted(last_mod),
        'pdf_hashes': sorted(hashes),
    }

# --- Backward compatibility shim ---
# Original signature: check_nice_alignment(synthesis, nice_guidelines)
# We keep it but make it a trivial keyword scan and measure latency.

def check_nice_alignment(synthesis: str, nice_guidelines: List[Dict]) -> Tuple[List[str], float]:
    start = time.time()
    matches = []
    low = (synthesis or '').lower()
    for g in (nice_guidelines or []):
        kw = str(g.get('keyword','')).lower()
        gid = str(g.get('id',''))
        if kw and kw in low:
            matches.append(gid)
    latency = time.time() - start
    return matches, latency
