import streamlit as st
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components

# --- Module imports (as in your project structure) ---
from modules.pubmed_retrieval import search_pubmed, fetch_abstracts
from modules.evidence_scoring import prioritise_documents
from modules.conflict_detection import detect_conflicts
from modules.attestation import generate_with_attestation
from modules.plotting import plot_crts_radar
from modules.logging import log_crts_both, log_human_feedback
from modules.guideline_checker import compute_ga_metrics, provenance_summary
from modules.guideline_retrieval import (
    align_claims_to_guidelines,
    process_uploaded_pdf,
    scrape_guideline_url,
)
from modules.crts import compute_crts

# NEW: Agent A – LLM-based evidence analyser
from agents.agent_evidence_analyser import EvidenceAnalyserAgent
from agents.agent_guideline_comparator import GuidelineComparatorAgent

st.set_page_config(page_title="Clinical RAG Active Verifiability", layout="wide")
st.title("Clinical RAG Active Verifiability Framework")
st.info("⚠️ Research decision-support tool. Not for clinical diagnosis or treatment.")

# ---------------------------------------
# Sidebar: Creator information
# ---------------------------------------
st.sidebar.markdown("**Inspired by:** Anthony Onoja")
st.sidebar.markdown("*School of Health Sciences, University of Surrey, UK*")
st.sidebar.markdown("[a.onoja@surrey.ac.uk](mailto:a.onoja@surrey.ac.uk)")
st.sidebar.divider()

# ---------------------------------------
# Sidebar: Framework methodology
# ---------------------------------------
st.sidebar.subheader("Framework Methodology")

with st.sidebar.expander("Active Verifiability (AV)", expanded=True):
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "figures", "Figure_RAGs_transparency.png")
    st.image(img_path, use_container_width=True)
    st.caption(
        "Figure 1. End-to-end architecture showing evidence ingestion, "
        "contradiction-aware retrieval, attestation-linked generation, "
        "guideline anchoring, and CRTS computation."
    )

with st.sidebar.expander("The CRTS Metric"):
    st.latex(r"CRTS = \alpha SF + \beta CRR + \gamma AR^{*} + \delta GA")
    st.write(
        "**SF**: % grounded • **CRR**: surfaced/detected dissent • "
        "**AR***: min(1,k/L) • **GA**: % claims aligned"
    )

# ---------------------------------------
# Sidebar: CRTS weights
# ---------------------------------------
st.sidebar.subheader("CRTS Weights")
alpha = st.sidebar.slider("α (SF)", 0.0, 1.0, 0.30, 0.05)
beta = st.sidebar.slider("β (CRR)", 0.0, 1.0, 0.30, 0.05)
gamma = st.sidebar.slider("γ (AR*)", 0.0, 1.0, 0.20, 0.05)
delta = st.sidebar.slider("δ (GA)", 0.0, 1.0, 0.20, 0.05)
w_sum = alpha + beta + gamma + delta
if w_sum == 0:
    alpha, beta, gamma, delta = 0.30, 0.30, 0.20, 0.20
else:
    alpha, beta, gamma, delta = (
        alpha / w_sum,
        beta / w_sum,
        gamma / w_sum,
        delta / w_sum,
    )

# ---------------------------------------
# Sidebar: Evidence timeframe filter
# ---------------------------------------
st.sidebar.subheader("Evidence Timeframe")

year_min, year_max = st.sidebar.slider(
    "Publication year range",
    min_value=1990,
    max_value=2026,
    value=(2015, 2026),
    help="Filter PubMed retrieval to publications within this year range.",
)
st.sidebar.caption(
    "Use this to give more weight to recent evidence or to explore historical guidance."
)

# ---------------------------------------
# Sidebar: LLM configuration + guidelines
# ---------------------------------------
st.sidebar.subheader("Configuration")
user_key = st.sidebar.text_input("Enter Gemini API Key (optional)", type="password")

# EvidenceAnalyserAgent and generate_with_attestation will both use this key if provided.

st.sidebar.subheader("Guideline Anchoring")
source_type = st.sidebar.radio("Guideline Source", ["Web Link", "Upload PDF"])

GUIDELINE_OPTIONS = {
    "NICE Oncology (NG14)": "https://www.nice.org.uk/guidance/ng14",
    "NHS Lanarkshire (PPPD)": "https://www.nhslanarkshire.scot.nhs.uk/services/physiotherapy/vestibular-physiotherapy/persistent-postural-perceptual-dizziness-pppd-or-3pd/",
    "Physio-pedia (PPPD)": "https://www.physio-pedia.com/Persistent_Postural-Perceptual_Dizziness",
    "Custom URL": ""
}

guideline_chunks = []
if source_type == "Web Link":
    selected_guideline = st.sidebar.selectbox("Select Guideline", list(GUIDELINE_OPTIONS.keys()))
    if selected_guideline == "Custom URL":
        guideline_url = st.sidebar.text_input(
            "Paste Guideline URL",
            value="",
            help="NICE/WHO or other trusted domain (allowlisted)",
        )
    else:
        guideline_url = GUIDELINE_OPTIONS[selected_guideline]

    if guideline_url:
        with st.sidebar:
            with st.spinner("Fetching web guideline..."):
                guideline_chunks = scrape_guideline_url(guideline_url)
        if guideline_chunks:
            st.sidebar.success(f"Loaded {len(guideline_chunks)} web segments.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload local protocol (PDF)", type="pdf")
    if uploaded_file:
        with st.sidebar:
            with st.spinner("Parsing local PDF..."):
                guideline_chunks = process_uploaded_pdf(uploaded_file)
        if guideline_chunks:
            st.sidebar.success(f"Loaded {len(guideline_chunks)} PDF pages.")

# ---------------------------------------
# Helper: estimate surfaced risks (now agent-informed)
# ---------------------------------------
def estimate_surfaced_risks_from_claims(
    evidence_claims, synthesis: str, detected_conflicts: int
) -> int:
    """
    Instead of regex on the synthesis alone, we use LLM-derived evidence_claims:
      - Count how many *risk-stance* claims appear explicitly in the synthesis.
      - Cap by detected_conflicts.
    If evidence_claims is empty, fall back to 0.
    """
    if detected_conflicts <= 0 or not synthesis or not evidence_claims:
        return 0

    risk_claims = [c for c in evidence_claims if c.get("stance") == "risk"]
    if not risk_claims:
        return 0

    surfaced = 0
    syn_lower = synthesis.lower()
    for c in risk_claims:
        # Transparent heuristic: see if the claim text or evidence_span appears in the synthesis
        txt = (c.get("claim_text") or "").strip()
        span = (c.get("evidence_span") or "").strip()
        if txt and txt[:40].lower() in syn_lower:
            surfaced += 1
        elif span and span[:40].lower() in syn_lower:
            surfaced += 1

    return min(surfaced, int(detected_conflicts))


# ---------------------------------------
# Main interaction
# ---------------------------------------

st.subheader("Enter a clinical research query")

# Example queries (non-limiting; just starting points)
example_queries = [
    "How effective is virtual reality as a rehabilitation technique for persistent postural-perceptual dizziness?",
    "How effective is VOR (vestibular occular reflex) training for persistent postural-perceptual dizziness?",
    "How effective is vestibular rehabilitation for persistent postural-perceptual dizziness?",
    "Do SGLT2 inhibitors reduce hospitalisation for heart failure?",
    "Is high-dose vitamin D safe in chronic kidney disease?",
    "How effective is cognitive behavioural therapy for adolescent social anxiety?",
]

with st.expander("Show example queries (click to paste)"):
    st.caption(
        "These examples illustrate different use cases (vestibular rehabilitation, "
        "cardiology, nephrology, mental health). You can also type any other "
        "clinical research question in the box below."
    )
    cols = st.columns(2)
    for i, q_example in enumerate(example_queries):
        col = cols[i % 2]
        if col.button(q_example, key=f"example_{i}"):
            st.session_state["query_text"] = q_example

# Text input bound to session_state so example buttons can populate it
query = st.text_input(
    "Clinical research question:",
    key="query_text",
    placeholder="e.g. Does SGLT2 inhibition improve renal outcomes in heart failure?",
)

if query:
    with st.spinner("Retrieving and analysing evidence..."):
        # 1) Retrieval + prioritisation
        pmids, effective_term = search_pubmed(
            query, year_min=year_min, year_max=year_max
        )
        if not pmids:
            st.warning(
                "No PubMed records were retrieved for this query "
                f"within {year_min}–{year_max}.\n\n"
                "Effective PubMed term:\n"
                f"`{effective_term}`\n\n"
                "Consider broadening the query or adjusting the timeframe."
            )
            st.stop()

        raw_docs = fetch_abstracts(pmids)
        documents = prioritise_documents(raw_docs, query)

        # 2) Conflict detection (document-level) – existing heuristic/module
        conflict_summary = detect_conflicts(documents)

        # 3) Agent A: LLM-based evidence & stance analysis (per abstract)
        evidence_agent = EvidenceAnalyserAgent(api_key=user_key or None)
        evidence_claims = evidence_agent.analyse_docs(documents)

        # 4) Generative synthesis + attestation (sentence-level grounding)
        synthesis, attestations = generate_with_attestation(
            query, documents, api_key=user_key
        )

        # 5) Guideline alignment (TF-IDF + cosine as before)
        claims = list(attestations.keys())
        guideline_alignment = align_claims_to_guidelines(claims, guideline_chunks)

        # 6) Conflict metrics for CRTS
        detected_D = int(conflict_summary.get("risk", 0))

        # Use LLM-derived evidence_claims to estimate surfaced risk signals in the synthesis
        surfaced_S = estimate_surfaced_risks_from_claims(
            evidence_claims, synthesis, detected_D
        )

        # 7) Agent B: Guideline comparator (richer labels) – optional
        guideline_agent = GuidelineComparatorAgent(
            api_key=user_key or None,
            default_source_name="Guideline",
        )
        guideline_assessments = guideline_agent.assess_claims(
            claims, guideline_chunks, top_k=3
        )

    # ------------------------
    # Layout columns
    # ------------------------
    col1, col2 = st.columns([2, 1])

    # ------------------------
    # Left: Synthesis, attestation, guideline anchoring
    # ------------------------
    with col1:
        st.subheader("Transparent Synthesis")
        st.write(synthesis)

        st.caption(
            f"Evidence timeframe: {year_min}–{year_max} "
            f"(filtered at retrieval using PubMed publication dates)."
        )

        with st.expander("Show PubMed search details"):
            st.markdown("**Effective PubMed term** (sent to Entrez.esearch):")
            st.code(effective_term, language="text")
            st.caption(
                "This is the exact term used for PubMed retrieval, including any synonym "
                "expansions (e.g. PPPD/3PD, vestibulo-ocular reflex, vestibular rehabilitation) "
                "and the selected year range."
            )

        st.subheader("Atomic Attestation Map")
        for claim, meta in attestations.items():
            with st.expander(f"Claim: {claim[:120]}..."):
                if isinstance(meta, dict) and "source_text" in meta:
                    st.success(f"**Source Text:** {meta['source_text']}")
                    st.caption(
                        f"**Document:** {meta.get('document_title')} (PMID: {meta.get('pmid')})"
                    )
                else:
                    st.warning(f"Source: {meta}")

        # Optional: show interpretive claims (no direct evidence) based on Agent A
        interpretive_claims = [
            c for c in evidence_claims if c.get("evidence_type") == "interpretive"
        ]
        if interpretive_claims:
            with st.expander("Interpretive / non-direct-evidence claims (Agent A)"):
                st.caption(
                    "These claims are inferred or interpretive according to the evidence analyser agent; "
                    "treat them with caution."
                )
                for c in interpretive_claims[:20]:
                    st.markdown(f"- **PMID {c['pmid']}** – {c['claim_text']}")

        st.subheader("Guideline Anchoring (Contextual Alignment)")
        for claim, match in guideline_alignment.items():
            if match:
                extra = []
                if "last_modified" in match and match["last_modified"]:
                    extra.append(f"Last-Modified: {match['last_modified']}")
                if "hash" in match:
                    extra.append(f"PDF hash: {match['hash']}")
                meta_txt = f" ({'; '.join(extra)})" if extra else ""
                st.write(
                    f"✅ **{claim[:80]}...** → {match['source']} "
                    f"({match['page']}, sim: {match['score']}){meta_txt}"
                )
            else:
                st.write(
                    f"❌ **{claim[:80]}...** → No explicit alignment in provided guideline."
                )

        ga, matched, total = compute_ga_metrics(guideline_alignment)
        st.caption(f"GA = {ga:.2f}  ({matched}/{total} claims aligned)")

        # Optional: richer alignment labels from Agent B
        if guideline_assessments:
            with st.expander(
                "Guideline comparator (Agent B): detailed alignment labels"
            ):
                st.caption(
                    "Agent B uses guideline excerpts and classifies each claim as "
                    "aligned, partially_aligned, contradicted, or not_addressed."
                )
                for ga_item in guideline_assessments[:50]:
                    label = ga_item["alignment_label"]
                    claim_text = ga_item["claim_text"]
                    src = ga_item["guideline_source"]
                    sec = ga_item["guideline_section"]
                    rationale = ga_item["rationale"]
                    st.markdown(
                        f"- **Claim:** {claim_text[:100]}...\n"
                        f"  - Alignment: `{label}`\n"
                        f"  - Source: {src} — {sec}\n"
                        f"  - Rationale: {rationale}"
                    )

        with st.expander("Guideline provenance"):
            prov = provenance_summary(guideline_alignment)
            if prov["sources"]:
                st.markdown("**Sources**")
                for s in prov["sources"]:
                    st.write("- ", s)
            if prov["last_modified"]:
                st.markdown("**Last-Modified (web)**")
                for lm in prov["last_modified"]:
                    st.write("- ", lm)
            if prov["pdf_hashes"]:
                st.markdown("**PDF hashes (local)**")
                for h in prov["pdf_hashes"]:
                    st.write("- ", h)

    # ------------------------
    # Right: Conflict summary + CRTS
    # ------------------------
    with col2:
        if conflict_summary.get("supportive", 0) or conflict_summary.get("risk", 0):
            st.warning("⚠️ Evidentiary dissent detected")
            st.write(f"Supportive studies: {conflict_summary.get('supportive', 0)}")
            st.write(f"Risk-signalling studies: {conflict_summary.get('risk', 0)}")
            fig, ax = plt.subplots()
            ax.bar(
                ["Supportive", "Risk-signalling"],
                [
                    conflict_summary.get("supportive", 0),
                    conflict_summary.get("risk", 0),
                ],
                color=["#2ecc71", "#e74c3c"],
            )
            ax.set_ylabel("Number of studies")
            ax.set_title(
                f"Evidence Balance (PubMed {year_min}–{year_max})"
            )
            st.pyplot(fig)

        crts = compute_crts(
            attestations=attestations,
            conflict_summary=conflict_summary,
            guideline_alignment=guideline_alignment,
            surfaced_risks_count=surfaced_S,
            k_seconds=5.0,
            weights={
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "delta": delta,
            },
        )

        st.divider()
        st.subheader("Clinical RAG Transparency Score (CRTS)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Source Fidelity", f"{crts['sf'] * 100:.0f}%")
        m2.metric("Conflict Reporting", f"{crts['crr'] * 100:.0f}%")
        m3.metric("Audit Responsiveness (AR*)", f"{crts['ar']:.2f}")
        m4.metric("Guideline Alignment", f"{crts['ga'] * 100:.0f}%")
        st.caption(
            "Audit latency L ≈ "
            f"{crts['L']:.0f} s/claim • weights α={crts['weights']['alpha']:.2f}, "
            f"β={crts['weights']['beta']:.2f}, γ={crts['weights']['gamma']:.2f}, "
            f"δ={crts['weights']['delta']:.2f}"
        )
        st.metric("Composite CRTS", f"{crts['crts']:.2f}")

        fig_radar = plot_crts_radar(
            {
                "Source Fidelity": crts["sf"],
                "Conflict Reporting": crts["crr"],
                "Audit Responsiveness": crts["ar"],
                "Guideline Alignment": crts["ga"],
            }
        )
        st.pyplot(fig_radar)

        # Log CSV + JSONL for audit
        log_crts_both(query, crts)

        st.divider()
        st.subheader("Clinician Validation Loop (HITL)")
        st.info("The clinician must stay in the loop to review the AI's reasoning.")
        with st.form("hitl_feedback_form"):
            val_rating = st.radio("How accurate and grounded was the synthesis?", [
                "Strongly Agree (Accurate & Contextual)",
                "Partially Agree (Minor inaccuracies)",
                "Disagree (Misaligned or Contradicted reasoning)",
                "Unable to assess"
            ])
            val_remarks = st.text_area("Clinician Comments / Rationale")
            val_submit = st.form_submit_button("Submit Validation")
            if val_submit:
                log_human_feedback(query, val_rating, val_remarks, synthesis)
                st.success("Human-in-the-loop reasoning verified and logged.")

    # ------------------------
    # Evidence Similarity Network
    # ------------------------
    st.divider()
    st.subheader("Evidence Similarity Network")
    try:
        from modules.evidence_network import build_evidence_network

        tags = conflict_summary.get("doc_tags", [])
        net = build_evidence_network(
            documents,
            query,
            similarity_threshold=0.25,
            conflict_doc_tags=tags,
        )
        path = "evidence_network.html"
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=500)
    except Exception as e:
        st.error(f"Network visualisation error: {e}")
