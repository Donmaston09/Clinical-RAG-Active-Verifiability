
# modules/attestation.py
"""
Transparent synthesis with claim-level attestation
- LLM path (if API key provided) with JSON contract and post-hoc validation
- Deterministic fallback (sentence extraction) preserving auditability
"""
from typing import Tuple, Dict, Any, List
import json
import re
import streamlit as st

try:
    import google.generativeai as genai
except Exception:  # allow running without the SDK (fallback still works)
    genai = None


def _split_sentences(text: str) -> List[str]:
    # Lightweight splitter; avoids external deps
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 30]


def _validate_json_claims(
    data: Dict[str, Any], docs_by_pmid: Dict[str, Dict]
) -> Tuple[str, Dict[str, Any]]:
    """
    Validate LLM JSON against source abstracts.
    - keep only claims whose source_text is a substring of the corresponding abstract
    - cap to 6 claims, deduplicate
    """
    synthesis = str(data.get("synthesis", "")).strip()
    att = data.get("attestations", {}) or {}
    clean_att = {}
    for claim, meta in att.items():
        pmid = str(meta.get("pmid", "")).strip()
        src_txt = str(meta.get("source_text", "")).strip()
        if not pmid or not src_txt or pmid not in docs_by_pmid:
            continue
        abstract = docs_by_pmid[pmid].get("abstract", "") or ""
        if src_txt and src_txt in abstract:
            c = claim if claim.endswith((".", "!", "?")) else claim + "."
            clean_att[c] = {
                "pmid": pmid,
                "source_text": src_txt,
                "document_title": docs_by_pmid[pmid].get("title"),
            }
    items = list(clean_att.items())[:6]
    return synthesis, dict(items)


def generate_with_attestation(
    query: str, documents: List[Dict[str, Any]], api_key: str = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesize clinical evidence and map atomic claims to exact source sentences.
    Returns (synthesis, attestations).
    Attestations schema: { claim_text: {pmid, source_text, document_title} }
    """
    docs_by_pmid = {str(d.get("pmid")): d for d in documents}

    # --- LLM PATH (if available) ---
    if api_key and genai is not None:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.5-flash-lite")

            # Build context lines WITHOUT f-strings in the final prompt
            context_lines = []
            for d in documents[:3]:
                pmid = str(d.get("pmid"))
                title = str(d.get("title") or "")
                abstract = (d.get("abstract", "") or "")[:900]
                context_lines.append(
                    "PMID:" + pmid + "\nTitle:" + title + "\nAbstract:" + abstract
                )
            context = "\n".join(context_lines)

            # Assemble prompt safely via list join (no triple quotes; no f-strings)
            prompt_lines = [
                "You are producing *auditable* outputs for clinical evidence synthesis.",
                "Extract 4–6 Atomic Claims (individual clinical facts) from the abstracts.",
                "For each claim, provide the exact sentence from the source abstract as 'source_text' and the correct 'pmid'.",
                "Return ONLY valid JSON (no markdown) with keys: synthesis, attestations.",
                "Example schema:",
                "{",
                '  "synthesis": "One or two sentences summarising the overall picture.",',
                '  "attestations": {',
                '     "<atomic claim text>": {"pmid": "<pmid>", "source_text": "<exact sentence>", "document_title": "<title>"}',
                "  }",
                "}",
                "Context:",
                context,
            ]
            prompt = "\n".join(prompt_lines)

            response = model.generate_content(prompt)
            raw = (getattr(response, "text", "") or "").strip()
            clean_json = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            synthesis, clean_att = _validate_json_claims(data, docs_by_pmid)
            if clean_att:
                return synthesis, clean_att
            else:
                st.sidebar.warning(
                    "LLM returned no verifiable claims; using deterministic fallback."
                )
        except Exception as e:
            if "429" in str(e):
                st.sidebar.warning("API quota reached. Using deterministic fallback.")
            else:
                st.sidebar.error(f"LLM error: {e}")

    # --- DETERMINISTIC FALLBACK ---
    attestations: Dict[str, Any] = {}
    synthesis_parts: List[str] = []
    for d in documents[:3]:
        abstract_text = d.get("abstract", "") or ""
        sents = _split_sentences(abstract_text)
        for s in sents[:2]:
            claim = s if s.endswith((".", "!", "?")) else s + "."
            if claim not in attestations:
                attestations[claim] = {
                    "pmid": str(d.get("pmid")),
                    "source_text": s,
                    "document_title": d.get("title"),
                }
        if sents:
            synthesis_parts.append(sents[0])
    synthesis = " ".join(synthesis_parts[:2]) + " (Deterministic Extraction)"
    return synthesis, attestations
