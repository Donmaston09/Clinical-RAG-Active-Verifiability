
"""
Evidence similarity network for visualising clusters and dissent.
- Cosine similarity over TF-IDF vectors of abstracts.
- Optional colouring by conflict tags (supportive/risk) if provided upstream.
- Deterministic, parameterised, and UI-friendly via PyVis.
"""
from typing import List, Dict, Optional
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

# Colours (consistent with app palette)
COL_SUPPORT = '#2ecc71'     # supportive (green)
COL_RISK = '#e74c3c'        # risk-signalling (red)
COL_BOTH = '#FF8C00'        # both supportive & risk (orange)
COL_NEUTRAL = '#95a5a6'     # neutral/unknown (grey)


def _node_colour(tag: Optional[Dict]) -> str:
    if not tag:
        return COL_NEUTRAL
    sup = bool(tag.get('supportive'))
    risk = bool(tag.get('risk'))
    if sup and risk:
        return COL_BOTH
    if risk:
        return COL_RISK
    if sup:
        return COL_SUPPORT
    return COL_NEUTRAL


def build_evidence_network(
    docs: List[Dict],
    query: str,
    similarity_threshold: float = 0.25,
    conflict_doc_tags: Optional[List[Dict]] = None,
) -> Network:
    """Builds an interactive evidence similarity network.

    Parameters
    ----------
    docs : list of dict
        Documents with keys: 'pmid', 'title', 'abstract', 'publication_type', 'year'.
    query : str
        The user query (shown in UI captions; not used in similarity).
    similarity_threshold : float
        Minimum cosine similarity for drawing an edge.
    conflict_doc_tags : list of dict or None
        Optional tags (same order as docs) with booleans 'supportive'/'risk'.

    Returns
    -------
    pyvis.network.Network
    """
    texts = [(d.get('abstract') or '') for d in docs]
    pmids = [str(d.get('pmid')) for d in docs]

    # Guard empty
    if not any(t.strip() for t in texts):
        net = Network(height='450px', width='100%', bgcolor='#ffffff', font_color='black')
        return net

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)

    G = nx.Graph()
    for i, pmid in enumerate(pmids):
        tag = conflict_doc_tags[i] if conflict_doc_tags and i < len(conflict_doc_tags) else None
        color = _node_colour(tag)
        title = docs[i].get('title', '')
        year = docs[i].get('year', '')
        ptypes = ', '.join(docs[i].get('publication_type', []) or [])
        tooltip = f"<b>PMID:</b> {pmid}<br><b>Title:</b> {title}<br><b>Year:</b> {year}<br><b>Type:</b> {ptypes}"
        if tag:
            st_terms = ', '.join(tag.get('support_terms', []) or [])
            rk_terms = ', '.join(tag.get('risk_terms', []) or [])
            tooltip += f"<br><b>Support terms:</b> {st_terms or '-'}<br><b>Risk terms:</b> {rk_terms or '-'}"
        G.add_node(pmid, label=f"PMID: {pmid}", title=tooltip, color=color)

    n = len(pmids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= similarity_threshold:
                # Edge weight conveys similarity strength
                G.add_edge(pmids[i], pmids[j], value=float(sim[i, j]))

    net = Network(height='450px', width='100%', bgcolor='#ffffff', font_color='black')
    net.from_nx(G)
    net.toggle_physics(True)
    return net
