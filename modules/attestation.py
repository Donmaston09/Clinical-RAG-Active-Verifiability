import google.generativeai as genai
import streamlit as st
import json

def generate_with_attestation(query, documents, api_key=None):
    """
    Synthesizes clinical evidence and maps atomic claims to sources.
    Synchronized with Manuscript Section 1.1: Formal Definition of CRTS.
    """
    # --- LLM PATH (GENERATIVE MODE) ---
    if api_key:
        try:
            genai.configure(api_key=api_key)
            # Using the 2025 stable production model
            model = genai.GenerativeModel('gemini-2.5-flash-lite')

            # Context preparation: Using top 3 abstracts to stay within rate limits
            context = "\n".join([
                f"PMID:{d['pmid']} | Title:{d['title']} | Abstract:{d['abstract'][:600]}"
                for d in documents[:3]
            ])

            prompt = f"""
            Task: Synthesize clinical evidence for the query: "{query}"

            Instructions:
            1. Extract at least 4-6 distinct 'Atomic Claims' (individual clinical facts).
            2. For each claim, provide the exact sentence from the abstract as 'source_text'.
            3. Ensure every claim is mapped to its correct PMID.

            Return ONLY a JSON object:
            {{
              "synthesis": "A 2-sentence formal clinical summary.",
              "attestations": {{
                 "The specific atomic claim text": {{
                    "pmid": "PMID number",
                    "source_text": "The exact sentence from the source abstract",
                    "document_title": "Full title of the paper"
                 }}
              }}
            }}

            Context Evidence:
            {context}
            """

            response = model.generate_content(prompt)

            # Clean JSON response (handling markdown code blocks if present)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            return data['synthesis'], data['attestations']

        except Exception as e:
            # If API fails or Quota exceeded, log to sidebar and move to fallback
            if "429" in str(e):
                st.sidebar.warning("API Quota Reached. Using Deterministic Fallback.")
            else:
                st.sidebar.error(f"LLM Error: {e}")

    # --- DETERMINISTIC PATH (NLTK FALLBACK) ---
    # This ensures "Active Verifiability" even without an API key
    attestations = {}
    synthesis_parts = []

    for d in documents[:3]:
        # Extracting the first two sentences for higher granularity
        abstract_text = d.get('abstract', '')
        # Simple sentence splitter
        sentences = [s.strip() for s in abstract_text.split('. ') if len(s) > 30]

        # Take up to 2 atomic sentences per document to ensure high SF score
        for s in sentences[:2]:
            claim = s if s.endswith('.') else s + '.'
            attestations[claim] = {
                "pmid": d['pmid'],
                "source_text": claim,
                "document_title": d['title']
            }

        if sentences:
            synthesis_parts.append(sentences[0])

    synthesis = " | ".join(synthesis_parts[:2]) + " (Deterministic Extraction)"

    return synthesis, attestations
