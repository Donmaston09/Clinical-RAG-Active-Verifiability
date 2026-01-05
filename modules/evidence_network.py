import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyvis.network import Network

def build_evidence_network(docs, query, similarity_threshold=0.25):
    texts = [d["abstract"] for d in docs]
    pmids = [d["pmid"] for d in docs]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    sim = cosine_similarity(X)

    # Use NetworkX for the logic
    G = nx.Graph()
    for i, pmid in enumerate(pmids):
        G.add_node(
            pmid,
            label=f"PMID: {pmid}",
            title=docs[i]["title"], # Hover text
            color="#3498db"
        )

    for i in range(len(pmids)):
        for j in range(i + 1, len(pmids)):
            if sim[i, j] > similarity_threshold:
                G.add_edge(pmids[i], pmids[j], value=sim[i, j])

    # Convert to Pyvis for the UI
    net = Network(height="450px", width="100%", bgcolor="#ffffff", font_color="black")
    net.from_nx(G)

    # Optional: Add physics for "bouncy" clinical evidence clusters
    net.toggle_physics(True)
    return net
