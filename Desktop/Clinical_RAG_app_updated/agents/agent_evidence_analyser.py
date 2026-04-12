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

from modules.llm_wrapper import generate_content as llm_generate

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

    def __init__(self, api_key: Optional[str] = None, provider: str = "Gemini"):
        self.api_key = api_key
        self.provider = provider

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
        Call the dynamic LLM model securely using the wrapper.
        """
        if not self.api_key:
            return ""

        try:
            return llm_generate(prompt, self.provider, self.api_key)
        except Exception as e:
            # Fail closed: do not throw, but return empty string
            print(f"Agent A LLM Error: {e}")
            return ""

    # -----------------------------
    # Public API
    # -----------------------------
    def analyse_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyse a list of PubMed-like documents and return a flat list of EvidenceClaim dicts.
        """
        results: List[Dict[str, Any]] = []

        if not self.api_key or not docs:
            # transparent no-op
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
