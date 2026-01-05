import requests
from bs4 import BeautifulSoup
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- NEW: DYNAMIC INGESTION HELPERS ---

def process_uploaded_pdf(uploaded_file):
    """Parses an uploaded PDF file from Streamlit's file_uploader."""
    guideline_chunks = []
    try:
        # We use io.BytesIO to handle the streamlit UploadedFile object
        with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text:
                    guideline_chunks.append({
                        "source": uploaded_file.name,
                        "page": i + 1,
                        "text": text
                    })
    except Exception as e:
        print(f"Error parsing uploaded PDF: {e}")
    return guideline_chunks

def scrape_guideline_url(url):
    """Fetches text from a clinical guideline URL (e.g., NICE, WHO)."""
    guideline_chunks = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Clean the HTML
        for script in soup(["script", "style"]):
            script.decompose()

        # Identify main content container
        main_content = soup.find('main') or soup.find('article') or soup.body

        # Split into pseudo-pages (by paragraphs) to maintain alignment granularity
        paragraphs = main_content.find_all(['p', 'li', 'h2', 'h3'])
        for i, p in enumerate(paragraphs):
            text = p.get_text().strip()
            if len(text) > 60:
                guideline_chunks.append({
                    "source": url,
                    "page": f"Section {i+1}",
                    "text": text
                })
    except Exception as e:
        print(f"Error scraping URL: {e}")
    return guideline_chunks

# --- UPDATED: ALIGNMENT LOGIC ---

def align_claims_to_guidelines(claims, guideline_chunks, threshold=0.15):
    """
    Computes semantic similarity between claims and guidelines.
    Synchronized with Manuscript Section: 'Guideline Anchoring'.
    """
    if not guideline_chunks or not claims:
        return {claim: None for claim in claims}

    # Vectorize the guideline text
    texts = [g["text"] for g in guideline_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    alignment = {}
    for claim in claims:
        claim_vec = vectorizer.transform([claim])
        sims = cosine_similarity(claim_vec, tfidf).flatten()

        best_idx = sims.argmax()
        best_score = sims[best_idx]

        if best_score >= threshold:
            g = guideline_chunks[best_idx]
            alignment[claim] = {
                "source": g["source"],
                "page": g["page"],
                "score": round(float(best_score), 2)
            }
        else:
            alignment[claim] = None

    return alignment
