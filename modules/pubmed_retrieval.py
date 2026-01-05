from Bio import Entrez
from config.settings import NCBI_EMAIL, MAX_PUBMED_RESULTS

Entrez.email = NCBI_EMAIL

def search_pubmed(query):
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=MAX_PUBMED_RESULTS,
        sort="relevance"
    )
    record = Entrez.read(handle)
    return record["IdList"]

def fetch_abstracts(pmids):
    if not pmids:
        return []

    handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="xml"
    )
    records = Entrez.read(handle)

    documents = []
    for article in records["PubmedArticle"]:
        try:
            medline = article["MedlineCitation"]
            article_data = medline["Article"]

            abstract = " ".join(article_data.get("Abstract", {}).get("AbstractText", []))
            title = article_data.get("ArticleTitle", "")
            pub_type = article_data.get("PublicationTypeList", [])
            year = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "")

            documents.append({
                "pmid": medline["PMID"],
                "title": title,
                "abstract": abstract,
                "publication_type": [str(p) for p in pub_type],
                "year": year
            })
        except Exception:
            continue

    return documents
