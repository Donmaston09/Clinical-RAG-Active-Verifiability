import streamlit as st
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from datetime import datetime

# -------------------------------------------------------------------
# Module imports
# -------------------------------------------------------------------
from modules.pubmed_retrieval import search_pubmed, fetch_abstracts
from modules.evidence_scoring import prioritise_documents
from modules.conflict_detection import detect_conflicts
from modules.attestation import generate_with_attestation
from modules.crts import compute_crts
from modules.plotting import plot_crts_radar
from modules.logging import log_crts
from modules.evidence_network import build_evidence_network
# Updated imports to include dynamic ingestion helpers
from modules.guideline_retrieval import (
    align_claims_to_guidelines,
    process_uploaded_pdf,
    scrape_guideline_url
)

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical RAG Active Verifiability",
    layout="wide"
)

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("Clinical RAG Active Verifiability Framework")
st.info("⚠️ Research decision-support tool. Not for clinical diagnosis or treatment.")

# -------------------------------------------------------------------
# Sidebar: Framework documentation
# -------------------------------------------------------------------
st.sidebar.subheader("Framework Methodology")

with st.sidebar.expander("Active Verifiability (AV)", expanded=True):
    st.image("figures/Figure_RAGs_transparency.png", use_container_width=True)
    st.caption(
        "Figure 1. End-to-end architecture showing evidence ingestion, "
        "contradiction-aware retrieval, attestation-linked generation, "
        "guideline anchoring, and CRTS computation."
    )

with st.sidebar.expander("The CRTS Metric"):
    st.latex(r"CRTS = \alpha \cdot SF + \beta \cdot CRR + \gamma \cdot AR + \delta \cdot GA")
    st.write(""" - **SF (Source Fidelity)**: % Grounded claims.
    - **CRR (Conflict Reporting)**: Ratio of surfaced dissent.
    - **AR (Audit Responsiveness)**: $1/L$ (Inverse Latency).
    - **GA (Guideline Alignment)**: Alignment with local/national protocols. """
    )

# -------------------------------------------------------------------
# Configuration & Dynamic Guideline Ingestion
# -------------------------------------------------------------------
st.sidebar.subheader("Configuration")

user_key = st.sidebar.text_input(
    "Enter Gemini API Key (optional)",
    type="password",
    help="Enables LLM-based atomic claim extraction"
)

st.sidebar.subheader("Guideline Anchoring")
source_type = st.sidebar.radio("Guideline Source", ["Web Link", "Upload PDF"])

guideline_chunks = []

if source_type == "Web Link":
    guideline_url = st.sidebar.text_input(
        "Paste Guideline URL",
        value="https://www.nice.org.uk/guidance/ng141",
        help="Paste a link to a NICE or WHO clinical guideline"
    )
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

# -------------------------------------------------------------------
# Main query
# -------------------------------------------------------------------
query = st.text_input("Enter a clinical research query:")

if query:
    with st.spinner("Retrieving and analysing evidence..."):

        # 1. PubMed retrieval and prioritisation
        pmids = search_pubmed(query)
        raw_docs = fetch_abstracts(pmids)
        documents = prioritise_documents(raw_docs, query)

        # 2. Conflict detection
        conflict_summary = detect_conflicts(documents)

        # 3. Transparent synthesis + atomic attestation
        synthesis, attestations = generate_with_attestation(
            query,
            documents,
            api_key=user_key
        )

        # 4. Dynamic Guideline Anchoring
        # Use the dynamic chunks from Sidebar (URL or Upload)
        claims = list(attestations.keys())
        guideline_alignment = align_claims_to_guidelines(claims, guideline_chunks)

        # Process matches for CRTS calculation
        nice_matches = [v for v in guideline_alignment.values() if v is not None]
        surfaced_risks = 1 if conflict_summary["detected"] else 0

    # ----------------------------------------------------------------
    # Layout
    # ----------------------------------------------------------------
    col1, col2 = st.columns([2, 1])

    # =========================
    # LEFT COLUMN
    # =========================
    with col1:
        st.subheader("Transparent Synthesis")
        st.write(synthesis)

        st.subheader("Atomic Attestation Map")
        for claim, meta in attestations.items():
            with st.expander(f"Claim: {claim[:120]}..."):
                if isinstance(meta, dict) and "source_text" in meta:
                    st.success(f"**Source Text:** {meta['source_text']}")
                    st.caption(
                        f"**Document:** {meta['document_title']} (PMID: {meta['pmid']})"
                    )
                else:
                    st.warning(f"Source: {meta}")

        st.subheader("Guideline Anchoring (Contextual Alignment)")
        for claim, match in guideline_alignment.items():
            if match:
                st.write(
                    f"✅ **{claim[:80]}...** → Aligned with *{match['source']}* "
                    f"({match['page']}, similarity: {match['score']})"
                )
            else:
                st.write(f"❌ **{claim[:80]}...** → No explicit alignment in provided guideline.")

    # =========================
    # RIGHT COLUMN
    # =========================
    with col2:
        if conflict_summary["detected"]:
            st.warning("⚠️ Evidentiary dissent detected")
            st.write(f"Supportive studies: {conflict_summary['supportive']}")
            st.write(f"Risk-signalling studies: {conflict_summary['risk']}")

            fig, ax = plt.subplots()
            ax.bar(
                ["Supportive", "Risk-signalling"],
                [conflict_summary["supportive"], conflict_summary["risk"]],
                color=['#2ecc71', '#e74c3c']
            )
            ax.set_ylabel("Number of studies")
            ax.set_title("Evidence Balance")
            st.pyplot(fig)

    # ----------------------------------------------------------------
    # CRTS computation
    # ----------------------------------------------------------------
    crts = compute_crts(
        attestations=attestations,
        conflict_summary=conflict_summary,
        surfaced_risks_count=surfaced_risks,
        nice_matches=nice_matches
    )

    st.divider()
    st.subheader("Clinical RAG Transparency Score (CRTS)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source Fidelity", f"{crts['sf']*100:.0f}%")
    m2.metric("Conflict Reporting", f"{crts['crr']*100:.0f}%")
    m3.metric("Audit Responsiveness", f"{crts['ar']:.2f}")
    m4.metric("Guideline Alignment", f"{crts['ga']*100:.0f}%")

    fig_radar = plot_crts_radar({
        "Source Fidelity": crts["sf"],
        "Conflict Reporting": crts["crr"],
        "Audit Responsiveness": crts["ar"],
        "Guideline Alignment": crts["ga"]
    })
    st.pyplot(fig_radar)

    # ----------------------------------------------------------------
    # Evidence network
    # ----------------------------------------------------------------
    st.divider()
    st.subheader("Evidence Similarity Network")

    try:
        net = build_evidence_network(documents, query)
        path = "evidence_network.html"
        net.save_graph(path)

        with open(path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=500)
    except Exception as e:
        st.error(f"Network visualisation error: {e}")

    # ----------------------------------------------------------------
    # Logging
    # ----------------------------------------------------------------
    log_crts(query, crts)
