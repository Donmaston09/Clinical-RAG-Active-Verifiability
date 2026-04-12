# modules/pubmed_retrieval.py

from Bio import Entrez
from config.settings import NCBI_EMAIL, MAX_PUBMED_RESULTS

Entrez.email = NCBI_EMAIL


def build_pubmed_term(user_query: str) -> str:
    """
    Build a more robust PubMed search term from the user query.
    - Handles PPPD / 3PD synonyms.
    - Handles VOR / vestibulo-ocular reflex spelling.
    - Handles vestibular rehabilitation terminology.

    This is intentionally conservative and transparent: it expands the query
    rather than replacing it entirely.
    """
    q = (user_query or "").lower()

    # Start with the raw user query as the base
    base = user_query

    # PPPD / 3PD condition synonyms
    if "persistent postural-perceptual dizziness" in q or "pppd" in q or "3pd" in q:
        cond = (
            '"persistent postural-perceptual dizziness"[tiab] '
            'OR "PPPD"[tiab] OR "3PD"[tiab]'
        )
    else:
        cond = None

    # VOR / vestibulo-ocular reflex
    vor_terms = None
    if (
        "vor" in q
        or "vestibular occular reflex" in q
        or "vestibulo ocular reflex" in q
        or "vestibulo-ocular reflex" in q
    ):
        vor_terms = '"vestibulo-ocular reflex"[tiab] OR "VOR"[tiab]'

    # Vestibular rehabilitation / physiotherapy
    vr_terms = None
    if (
        "vestibular rehabilitation" in q
        or "vestibular physio" in q
        or "vestibular physiotherapy" in q
        or "vestibular therapy" in q
    ):
        vr_terms = (
            '"vestibular rehabilitation"[tiab] OR "vestibular physiotherapy"[tiab] '
            'OR "vestibular therapy"[tiab]'
        )

    # Build expanded term transparently
    parts = []

    # If we have a condition synonym, use that; otherwise keep the original query
    if cond:
        parts.append(f"({cond})")
    else:
        parts.append(f"({user_query})")

    # Add VOR or vestibular rehab terms if present
    extra_terms = []
    if vor_terms:
        extra_terms.append(f"({vor_terms})")
    if vr_terms:
        extra_terms.append(f"({vr_terms})")

    if extra_terms:
        parts.append(" AND (" + " OR ".join(extra_terms) + ")")

    return " ".join(parts)


def search_pubmed(query: str, year_min: int = None, year_max: int = None):
    """
    Search PubMed using Entrez.esearch with:
      - robust term building (PPPD/VOR/vestibular rehab aware),
      - optional year range filtering.

    Returns: (pmid_list, effective_term)
    """
    term = build_pubmed_term(query)

    esearch_kwargs = {
        "db": "pubmed",
        "term": term,
        "retmax": MAX_PUBMED_RESULTS,
        "sort": "relevance",
    }

    # Year range (if provided)
    if year_min is not None and year_max is not None:
        esearch_kwargs["mindate"] = str(year_min)
        esearch_kwargs["maxdate"] = str(year_max)
        esearch_kwargs["datetype"] = "pdat"  # publication date

    handle = Entrez.esearch(**esearch_kwargs)
    record = Entrez.read(handle)
    pmids = record.get("IdList", [])
    return pmids, term


def fetch_abstracts(pmids):
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml",
    )
    records = Entrez.read(handle)

    documents = []
    for article in records.get("PubmedArticle", []):
        try:
            medline = article["MedlineCitation"]
            article_data = medline["Article"]

            abstract = " ".join(
                article_data.get("Abstract", {}).get("AbstractText", [])
            )
            title = article_data.get("ArticleTitle", "")
            pub_type = article_data.get("PublicationTypeList", [])
            pub_date = (
                article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            )
            year = pub_date.get("Year", "")

            documents.append(
                {
                    "pmid": medline["PMID"],
                    "title": title,
                    "abstract": abstract,
                    "publication_type": [str(p) for p in pub_type],
                    "year": year,
                }
            )
        except Exception:
            continue

    return documents
