import json
from typing import List, Dict, Any, Optional

from modules.attestation import generate_with_attestation  # reuse your existing LLM client / API setup


# agents/agent_evidence_analyser.py

"""
Agent A: LLM-based evidence analyser

- For each PubMed-style document (PMID, title, abstract),
  extract atomic clinical claims with stance, evidence type, and outcome type.
- Uses Google Gemini directly (same SDK as modules/attestation.py),
  but in a separate, clearly-auditable pathway.
"""

from typing import List, Dict, Any, Optional
import json

try:
    import google.generativeai as genai
except Exception:
    genai = None


class EvidenceAnalyserAgent:
    """
    Agent A: For each PubMed abstract, extract atomic claims and label:
      - stance: supportive | risk | neutral | inconclusive
      - evidence_type: direct_evidence | interpretive
      - outcome_type: efficacy | safety | other | unclear

    Returns a flat list of EvidenceClaim dicts:
        {
            "pmid": str,
            "claim_id": str,
            "claim_text": str,
            "stance": str,
            "evidence_span": str,
            "evidence_type": str,
            "outcome_type": str,
            "reasoning": str,
        }
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash-lite"):
        """
        api_key: Gemini API key. If None or SDK unavailable, the agent will
                 return an empty list (transparent no-op) rather than fail.
        model_name: Gemini model name, aligned with your attestation path.
        """
        self.api_key = api_key
        self.model_name = model_name

        self._model = None
        if genai is not None and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name)
            except Exception:
                # Fail gracefully: treat as unavailable
                self._model = None

    # -----------------------------
    # Prompt building
    # -----------------------------
    @staticmethod
    def _build_prompt_for_doc(doc: Dict[str, Any]) -> str:
        """
        Build an instruction prompt for a single document.

        Expected doc keys:
          - 'pmid'
          - 'title'
          - 'abstract'
        """
        pmid = doc.get("pmid")
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")

        lines = [
            "You are assisting in a clinical evidence transparency system.",
            "Your task is to read the following biomedical abstract and extract a small set of atomic clinical claims.",
            "",
            "For each claim, you MUST provide:",
            '  - "claim_text": the atomic statement (a single clinical proposition),',
            '  - "stance": one of: "supportive", "risk", "neutral", or "inconclusive",',
            '  - "evidence_span": the exact sentence(s) from the abstract supporting this claim,',
            '  - "evidence_type": "direct_evidence" if the abstract directly reports data about this claim,',
            '                     or "interpretive" if the claim is inferred or speculative,',
            '  - "outcome_type": one of: "efficacy", "safety", "other", or "unclear",',
            '  - "reasoning": a brief explanation of why you chose this stance and evidence_type.',
            "",
            "Abstract details:",
            f"PMID: {pmid}",
            f"Title: {title}",
            "Abstract:",
            abstract,
            "",
            "Extract between 1 and 5 clinically relevant claims.",
            "",
            "Return ONLY a JSON list of objects with the fields above.",
            "Example:",
            '[{"claim_text": "...", "stance": "risk", "evidence_span": "...", '
            '"evidence_type": "direct_evidence", "outcome_type": "safety", "reasoning": "..."}]',
        ]
        return "\n".join(lines)

    # -----------------------------
    # Low-level LLM call
    # -----------------------------
    def _call_llm(self, prompt: str) -> str:
        """
        Call Gemini with the provided prompt and return raw text.

        If the model or API key is not available, return an empty string,
        making the agent a safe no-op.
        """
        if self._model is None:
            return ""

        try:
            response = self._model.generate_content(prompt)
            raw = (getattr(response, "text", "") or "").strip()
            return raw
        except Exception:
            # Fail closed: do not throw, but return empty string
            return ""

    # -----------------------------
    # Public API
    # -----------------------------
    def analyse_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyse a list of PubMed-like documents and return a flat list of EvidenceClaim dicts.
        """
        results: List[Dict[str, Any]] = []

        if self._model is None or not docs:
            # No LLM configured or no documents; transparent no-op
            return results

        for doc in docs:
            pmid = str(doc.get("pmid") or "").strip()
            abstract = (doc.get("abstract") or "").strip()
            if not pmid or not abstract:
                continue

            prompt = self._build_prompt_for_doc(doc)
            raw = self._call_llm(prompt)
            if not raw:
                continue

            # Strip any accidental code fencing
            clean = raw.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(clean)
            except json.JSONDecodeError:
                # Model did not follow JSON contract; skip this doc
                continue

            if not isinstance(parsed, list):
                continue

            for idx, c in enumerate(parsed, start=1):
                claim_text = (c.get("claim_text") or "").strip()
                if not claim_text:
                    continue

                stance = (c.get("stance") or "neutral").strip().lower()
                if stance not in {"supportive", "risk", "neutral", "inconclusive"}:
                    stance = "neutral"

                evidence_type = (c.get("evidence_type") or "interpretive").strip().lower()
                if evidence_type not in {"direct_evidence", "interpretive"}:
                    evidence_type = "interpretive"

                outcome_type = (c.get("outcome_type") or "unclear").strip().lower()
                if outcome_type not in {"efficacy", "safety", "other", "unclear"}:
                    outcome_type = "unclear"

                evidence_span = (c.get("evidence_span") or "").strip()
                reasoning = (c.get("reasoning") or "").strip()

                claim = {
                    "pmid": pmid,
                    "claim_id": f"{pmid}-{idx}",
                    "claim_text": claim_text,
                    "stance": stance,
                    "evidence_span": evidence_span,
                    "evidence_type": evidence_type,
                    "outcome_type": outcome_type,
                    "reasoning": reasoning,
                }
                results.append(claim)

        return results
