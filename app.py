
import streamlit as st
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import re

# --- Module imports (as in your project structure) ---
from modules.pubmed_retrieval import search_pubmed, fetch_abstracts
from modules.evidence_scoring import prioritise_documents
from modules.conflict_detection import detect_conflicts
from modules.attestation import generate_with_attestation
from modules.plotting import plot_crts_radar
from modules.logging import log_crts_both
from modules.guideline_checker import compute_ga_metrics, provenance_summary
from modules.guideline_retrieval import align_claims_to_guidelines, process_uploaded_pdf, scrape_guideline_url
from modules.crts import compute_crts

st.set_page_config(page_title="Clinical RAG Active Verifiability", layout="wide")
st.title("Clinical RAG Active Verifiability Framework")
st.info("⚠️ Research decision-support tool. Not for clinical diagnosis or treatment.")

st.sidebar.subheader("Framework Methodology")

with st.sidebar.expander("Active Verifiability (AV)", expanded=True):
    st.image("figures/Figure_RAGs_transparency.png", use_container_width=True)
    st.caption(
        "Figure 1. End-to-end architecture showing evidence ingestion, "
        "contradiction-aware retrieval, attestation-linked generation, "
        "guideline anchoring, and CRTS computation."
    )
with st.sidebar.expander("The CRTS Metric"):
    st.latex(r"CRTS = \alpha SF + \beta CRR + \gamma AR^{*} + \delta GA")
    st.write("**SF**: % grounded • **CRR**: surfaced/detected dissent • **AR***: min(1,k/L) • **GA**: % claims aligned")

st.sidebar.subheader("CRTS Weights")
alpha = st.sidebar.slider("α (SF)", 0.0, 1.0, 0.30, 0.05)
beta = st.sidebar.slider("β (CRR)", 0.0, 1.0, 0.30, 0.05)
gamma = st.sidebar.slider("γ (AR*)", 0.0, 1.0, 0.20, 0.05)
delta = st.sidebar.slider("δ (GA)", 0.0, 1.0, 0.20, 0.05)
w_sum = alpha + beta + gamma + delta
if w_sum == 0:
    alpha, beta, gamma, delta = 0.30, 0.30, 0.20, 0.20
else:
    alpha, beta, gamma, delta = (alpha/w_sum, beta/w_sum, gamma/w_sum, delta/w_sum)

st.sidebar.subheader("Configuration")
user_key = st.sidebar.text_input("Enter Gemini API Key (optional)", type="password")
st.sidebar.subheader("Guideline Anchoring")
source_type = st.sidebar.radio("Guideline Source", ["Web Link", "Upload PDF"])
DEFAULT_NICE_URL = "https://www.nice.org.uk/guidance/ng14"

guideline_chunks = []
if source_type == "Web Link":
    guideline_url = st.sidebar.text_input("Paste Guideline URL", value=DEFAULT_NICE_URL, help="NICE/WHO or other trusted domain (allowlisted)")
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

RISK_TERMS = re.compile(r"(risk|toxicit|contraindicat|adverse|harm|warning|black box)", re.I)

def estimate_surfaced_risks(synthesis: str, detected_conflicts: int) -> int:
    if not synthesis or detected_conflicts <= 0:
        return 0
    sents = re.split(r"(?<=[.!?])\s+", synthesis)
    count = sum(1 for s in sents if RISK_TERMS.search(s))
    return min(count, int(detected_conflicts))

query = st.text_input("Enter a clinical research query:")
if query:
    with st.spinner("Retrieving and analysing evidence..."):
        pmids = search_pubmed(query)
        raw_docs = fetch_abstracts(pmids)
        documents = prioritise_documents(raw_docs, query)
        conflict_summary = detect_conflicts(documents)
        synthesis, attestations = generate_with_attestation(query, documents, api_key=user_key)
        claims = list(attestations.keys())
        guideline_alignment = align_claims_to_guidelines(claims, guideline_chunks)
        detected_D = int(conflict_summary.get("risk", 0))
        surfaced_S = estimate_surfaced_risks(synthesis, detected_D)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Transparent Synthesis")
        st.write(synthesis)
        st.subheader("Atomic Attestation Map")
        for claim, meta in attestations.items():
            with st.expander(f"Claim: {claim[:120]}..."):
                if isinstance(meta, dict) and "source_text" in meta:
                    st.success(f"**Source Text:** {meta['source_text']}")
                    st.caption(f"**Document:** {meta.get('document_title')} (PMID: {meta.get('pmid')})")
                else:
                    st.warning(f"Source: {meta}")
        st.subheader("Guideline Anchoring (Contextual Alignment)")
        for claim, match in guideline_alignment.items():
            if match:
                extra = []
                if 'last_modified' in match and match['last_modified']:
                    extra.append(f"Last-Modified: {match['last_modified']}")
                if 'hash' in match:
                    extra.append(f"PDF hash: {match['hash']}")
                meta_txt = f" ({'; '.join(extra)})" if extra else ""
                st.write(f"✅ **{claim[:80]}...** → {match['source']} ({match['page']}, sim: {match['score']}){meta_txt}")
            else:
                st.write(f"❌ **{claim[:80]}...** → No explicit alignment in provided guideline.")
        ga, matched, total = compute_ga_metrics(guideline_alignment)
        st.caption(f"GA = {ga:.2f}  ({matched}/{total} claims aligned)")
        with st.expander("Guideline provenance"):
            prov = provenance_summary(guideline_alignment)
            if prov['sources']:
                st.markdown("**Sources**")
                for s in prov['sources']:
                    st.write("- ", s)
            if prov['last_modified']:
                st.markdown("**Last-Modified (web)**")
                for lm in prov['last_modified']:
                    st.write("- ", lm)
            if prov['pdf_hashes']:
                st.markdown("**PDF hashes (local)**")
                for h in prov['pdf_hashes']:
                    st.write("- ", h)

    with col2:
        if conflict_summary.get("supportive", 0) or conflict_summary.get("risk", 0):
            st.warning("⚠️ Evidentiary dissent detected")
            st.write(f"Supportive studies: {conflict_summary.get('supportive', 0)}")
            st.write(f"Risk-signalling studies: {conflict_summary.get('risk', 0)}")
            fig, ax = plt.subplots()
            ax.bar(["Supportive", "Risk-signalling"],
                   [conflict_summary.get("supportive", 0), conflict_summary.get("risk", 0)],
                   color=['#2ecc71', '#e74c3c'])
            ax.set_ylabel("Number of studies")
            ax.set_title("Evidence Balance")
            st.pyplot(fig)

        crts = compute_crts(
            attestations=attestations,
            conflict_summary=conflict_summary,
            guideline_alignment=guideline_alignment,
            surfaced_risks_count=surfaced_S,
            k_seconds=5.0,
            weights={"alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta}
        )
        st.divider()
        st.subheader("Clinical RAG Transparency Score (CRTS)")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Source Fidelity", f"{crts['sf']*100:.0f}%")
        m2.metric("Conflict Reporting", f"{crts['crr']*100:.0f}%")
        m3.metric("Audit Responsiveness (AR*)", f"{crts['ar']:.2f}")
        m4.metric("Guideline Alignment", f"{crts['ga']*100:.0f}%")
        st.caption(f"Audit latency L ≈ {crts['L']:.0f} s/claim • weights α={crts['weights']['alpha']:.2f}, β={crts['weights']['beta']:.2f}, γ={crts['weights']['gamma']:.2f}, δ={crts['weights']['delta']:.2f}")
        st.metric("Composite CRTS", f"{crts['crts']:.2f}")
        fig_radar = plot_crts_radar({
            "Source Fidelity": crts["sf"],
            "Conflict Reporting": crts["crr"],
            "Audit Responsiveness": crts["ar"],
            "Guideline Alignment": crts["ga"],
        })
        st.pyplot(fig_radar)

        # Log CSV + JSONL
        log_crts_both(query, crts)

    st.divider()
    st.subheader("Evidence Similarity Network")
    try:
        from modules.evidence_network import build_evidence_network
        tags = conflict_summary.get("doc_tags", [])
        net = build_evidence_network(documents, query, similarity_threshold=0.25, conflict_doc_tags=tags)
        path = "evidence_network.html"
        net.save_graph(path)
        with open(path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=500)
    except Exception as e:
        st.error(f"Network visualisation error: {e}")
