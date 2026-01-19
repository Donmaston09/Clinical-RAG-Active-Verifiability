
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import pdfplumber
import io
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Provenance & Safety: allowlist domains for live guideline scraping ---
ALLOWED_GUIDELINE_DOMAINS = {
    'www.nice.org.uk', 'nice.org.uk',
    'www.who.int', 'who.int',
    'www.gov.uk', 'gov.uk',
    'www.nhs.uk', 'nhs.uk',
    'ecdc.europa.eu', 'www.ecdc.europa.eu'
}


def process_uploaded_pdf(uploaded_file):
    """Parses an uploaded PDF file from Streamlit's file_uploader and returns
    a list of chunks with provenance (filename, page, sha256 hash) for alignment.
    """
    guideline_chunks = []
    try:
        raw = uploaded_file.read()
        file_hash = hashlib.sha256(raw).hexdigest()[:12]
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    guideline_chunks.append({
                        "source": uploaded_file.name,
                        "page": i + 1,
                        "text": text,
                        "hash": file_hash
                    })
    except Exception as e:
        print(f"Error parsing uploaded PDF: {e}")
    return guideline_chunks


def _domain_allowed(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        # Strip port if present
        host = host.split(':')[0]
        return host in ALLOWED_GUIDELINE_DOMAINS
    except Exception:
        return False


def scrape_guideline_url(url):
    """Fetches text from a clinical guideline URL (e.g., NICE, WHO) with a domain
    allowlist to mitigate spoofing/typosquatting and captures Last-Modified headers.
    """
    guideline_chunks = []
    try:
        if not _domain_allowed(url):
            raise ValueError("URL domain not in allowlist. Provide a trusted guideline URL or upload a PDF.")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        last_mod = response.headers.get('Last-Modified')
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.body
        paragraphs = main_content.find_all(['p', 'li', 'h2', 'h3']) if main_content else []
        for i, p in enumerate(paragraphs):
            text = p.get_text().strip()
            if len(text) > 60:
                guideline_chunks.append({
                    "source": url,
                    "page": f"Section {i+1}",
                    "text": text,
                    "last_modified": last_mod
                })
    except Exception as e:
        print(f"Error scraping URL: {e}")
    return guideline_chunks


# --- Alignment logic (TF-IDF, cosine); threshold consistent with manuscript ---
def align_claims_to_guidelines(claims, guideline_chunks, threshold=0.15):
    """
    Computes similarity between claims and guideline chunks.
    Returns dict: claim -> {source, page, score, meta...} or None.
    """
    if not guideline_chunks or not claims:
        return {claim: None for claim in claims}

    texts = [g["text"] for g in guideline_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    alignment = {}
    for claim in claims:
        claim_vec = vectorizer.transform([claim])
        sims = cosine_similarity(claim_vec, tfidf).flatten()
        best_idx = sims.argmax()
        best_score = float(sims[best_idx])
        if best_score >= threshold:
            g = guideline_chunks[best_idx]
            # Keep provenance metadata if present
            meta = {k: v for k, v in g.items() if k not in {"text"}}
            alignment[claim] = {
                "source": g.get("source"),
                "page": g.get("page"),
                "score": round(best_score, 2),
                **meta
            }
        else:
            alignment[claim] = None
    return alignment
