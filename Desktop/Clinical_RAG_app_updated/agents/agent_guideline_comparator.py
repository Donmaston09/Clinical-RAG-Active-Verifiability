# agents/agent_guideline_comparator.py

"""
Agent B: Guideline Comparator

For each generated claim, compare it against top-k guideline chunks and classify:

  alignment_label ∈ {"aligned", "partially_aligned", "contradicted", "not_addressed"}

The agent uses:
  - Your existing TF-IDF-based guideline retrieval (claim -> top-k chunks)
  - Google Gemini for the final judgement, returning structured, auditable outputs.

It does NOT change your GA/CRTS computation; it provides an additional,
richer view you can surface in the UI and later decide how to integrate.
"""

from typing import List, Dict, Any, Optional
import json

try:
    import google.generativeai as genai
except Exception:
    genai = None

# If you prefer, you can place this helper in modules/guideline_retrieval_extended
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def top_guideline_chunks_for_claims(
    claims: List[str],
    guideline_chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    For each claim string, return a list of up to top_k guideline chunk dicts,
    each annotated with a 'similarity_score' field.

    guideline_chunks: list of dicts, each with at least:
        - 'text'
        - 'source'
        - 'section'
    """
    out: Dict[str, List[Dict[str, Any]]] = {c: [] for c in claims}
    if not claims or not guideline_chunks:
        return out

    texts = [gc.get("text", "") for gc in guideline_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    X_guidelines = vectorizer.fit_transform(texts)
    X_claims = vectorizer.transform(claims)

    sim_matrix = cosine_similarity(X_claims, X_guidelines)

    for i, claim in enumerate(claims):
        sims = sim_matrix[i]
        top_indices = sims.argsort()[::-1][:top_k]
        entries: List[Dict[str, Any]] = []
        for idx in top_indices:
            gc = dict(guideline_chunks[idx])  # shallow copy
            gc["similarity_score"] = float(sims[idx])
            entries.append(gc)
        out[claim] = entries

    return out


class GuidelineComparatorAgent:
    """
    Agent B: takes claims and guideline chunks, labels guideline alignment.

    Outputs a list of GuidelineAssessment objects, each with:
        {
            "claim_text": str,
            "guideline_source": str,
            "guideline_section": str,
            "guideline_excerpt": str,
            "alignment_label": str,
            "similarity_score": float,
            "rationale": str,
        }
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash-lite",
        default_source_name: str = "Guideline",
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.default_source_name = default_source_name

        self._model = None
        if genai is not None and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name)
            except Exception:
                self._model = None

    # -----------------------------
    # Prompt building
    # -----------------------------
    def _build_prompt(
        self,
        claim_text: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "You are assisting in a clinical evidence transparency system.",
            "Your task is to assess how a clinical claim relates to guideline excerpts.",
            "",
            "Clinical claim:",
            claim_text,
            "",
            "Relevant guideline excerpts:",
        ]
        for i, gc in enumerate(candidates, start=1):
            src = gc.get("source", self.default_source_name)
            sec = gc.get("section", "")
            txt = gc.get("text", "")
            score = gc.get("similarity_score", 0.0)
            lines.extend(
                [
                    f"--- Guideline {i} ---",
                    f"Source: {src}",
                    f"Section: {sec}",
                    f"Similarity_score: {score:.3f}",
                    "Excerpt:",
                    txt,
                    "",
                ]
            )

        lines.extend(
            [
                "You MUST choose a single alignment_label from one of:",
                '"aligned", "partially_aligned", "contradicted", "not_addressed".',
                "",
                "Definitions:",
                '- "aligned": The guideline clearly supports or describes this claim.',
                '- "partially_aligned": The guideline touches on the topic but only partially supports the exact claim.',
                '- "contradicted": The guideline recommends against or contradicts the claim.',
                '- "not_addressed": The guideline does not clearly address this claim.',
                "",
                "Return ONLY a single JSON object with the keys:",
                '{',
                '  "alignment_label": "aligned" | "partially_aligned" | "contradicted" | "not_addressed",',
                '  "chosen_source": str,',
                '  "chosen_section": str,',
                '  "chosen_excerpt": str,',
                '  "rationale": str',
                '}',
            ]
        )
        return "\n".join(lines)

    # -----------------------------
    # LLM call
    # -----------------------------
    def _call_llm(self, prompt: str) -> str:
        if self._model is None:
            return ""
        try:
            response = self._model.generate_content(prompt)
            raw = (getattr(response, "text", "") or "").strip()
            return raw
        except Exception:
            return ""

    # -----------------------------
    # Public API
    # -----------------------------
    def assess_claims(
        self,
        claims: List[str],
        guideline_chunks: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Given a list of claim texts and guideline chunks, return a list of assessments.
        """
        results: List[Dict[str, Any]] = []

        if self._model is None or not claims or not guideline_chunks:
            return results

        # Pre-filter: TF-IDF-based nearest guideline chunks per claim
        claim_to_chunks = top_guideline_chunks_for_claims(
            claims, guideline_chunks, top_k=top_k
        )

        for claim_text in claims:
            candidates = claim_to_chunks.get(claim_text, []) or []

            # If TF-IDF found nothing (unlikely), treat as not_addressed
            if not candidates:
                results.append(
                    {
                        "claim_text": claim_text,
                        "guideline_source": self.default_source_name,
                        "guideline_section": "",
                        "guideline_excerpt": "",
                        "alignment_label": "not_addressed",
                        "similarity_score": 0.0,
                        "rationale": "No relevant guideline chunk identified by TF-IDF pre-filter.",
                    }
                )
                continue

            prompt = self._build_prompt(claim_text, candidates)
            raw = self._call_llm(prompt)
            if not raw:
                # LLM failure; treat as not_addressed
                gc0 = candidates[0]
                results.append(
                    {
                        "claim_text": claim_text,
                        "guideline_source": gc0.get("source", self.default_source_name),
                        "guideline_section": gc0.get("section", ""),
                        "guideline_excerpt": gc0.get("text", "")[:500],
                        "alignment_label": "not_addressed",
                        "similarity_score": float(gc0.get("similarity_score", 0.0)),
                        "rationale": "LLM error or empty response; treated as not_addressed.",
                    }
                )
                continue

            clean = raw.replace("```json", "").replace("```", "").strip()
            try:
                obj = json.loads(clean)
            except json.JSONDecodeError:
                # Bad JSON; fallback
                gc0 = candidates[0]
                results.append(
                    {
                        "claim_text": claim_text,
                        "guideline_source": gc0.get("source", self.default_source_name),
                        "guideline_section": gc0.get("section", ""),
                        "guideline_excerpt": gc0.get("text", "")[:500],
                        "alignment_label": "not_addressed",
                        "similarity_score": float(gc0.get("similarity_score", 0.0)),
                        "rationale": "LLM returned non-JSON; treated as not_addressed.",
                    }
                )
                continue

            alignment_label = (obj.get("alignment_label") or "not_addressed").strip().lower()
            if alignment_label not in {
                "aligned",
                "partially_aligned",
                "contradicted",
                "not_addressed",
            }:
                alignment_label = "not_addressed"

            gc0 = candidates[0]
            assessment = {
                "claim_text": claim_text,
                "guideline_source": obj.get(
                    "chosen_source", gc0.get("source", self.default_source_name)
                ),
                "guideline_section": obj.get("chosen_section", gc0.get("section", "")),
                "guideline_excerpt": obj.get("chosen_excerpt", gc0.get("text", "")[:500]),
                "alignment_label": alignment_label,
                "similarity_score": float(gc0.get("similarity_score", 0.0)),
                "rationale": obj.get("rationale", ""),
            }
            results.append(assessment)

        return results