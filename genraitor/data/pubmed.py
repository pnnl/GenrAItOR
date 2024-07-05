"""Pubmed utility functions."""

from pathlib import Path
from typing import Generator

import duckdb
import pandas as pd
from Bio import Entrez
from metapub import PubMedFetcher

from ..conf import log

Entrez.email = "matt.jensen@pnnl.gov"

uniprot_ids = ["P12345", "Q67890"]


CACHE_DIR = str(Path.home() / ".cache" / "genraitor")


def _fetch_articles(pubmed_ids, cache_dir=CACHE_DIR) -> Generator:
    fetcher = PubMedFetcher(cachedir=cache_dir)

    n_articles = len(pubmed_ids)
    for i, (uniprot, pubmed) in enumerate(pubmed_ids.itertuples(index=False)):
        log.info(f"fetching: {pubmed} ({i} / {n_articles})")
        article = fetcher.article_by_pmid(pubmed)
        yield uniprot, pubmed, article


def articles(pubmed_ids):
    """Fetch pubmed abstracts given a list of ids."""
    results = []
    for uniprot_id, pubmed_id, article in _fetch_articles(pubmed_ids):
        result = {}
        result["uniprot_id"] = uniprot_id
        result["pubmed_id"] = pubmed_id
        result["doi"] = article.doi
        result["title"] = article.title
        result["abstract"] = article.abstract
        results.append(result)
    return pd.DataFrame(results)


def fetch_pubmed_records(uniprot_id):
    query = f"'{uniprot_id}'[WORD]"
    uniprot_id

    with Entrez.esearch(db="pubmed", term=query) as handle:
        record = Entrez.read(handle)
    record

    return record


def fetch_pubmed_documents(pubmed_ids):
    handle = Entrez.efetch(
        db="pubmed",
        id=pubmed_ids,
        rettype="medline",
        retmode="text",
    )
    documents = handle.read()
    handle.close()
    return documents


def docs(uniprot_ids):
    breakpoint()

    for uniprot_id in uniprot_ids:
        break

        pubmed_ids = fetch_pubmed_records(uniprot_id)
        if pubmed_ids:
            documents = fetch_pubmed_documents(pubmed_ids)
            print(f"Documents for UniProt ID {uniprot_id}:\n{documents}\n")
        else:
            print(f"No PubMed documents found for UniProt ID {uniprot_id}")


def ids_from_uniprot(uniprot_data):
    pubmed_ids = duckdb.sql("""
        SELECT
            uniprot_id
            ,source_id
        from uniprot_data
        WHERE source_id is not null
        AND source_name = 'PubMed'
        GROUP BY
            uniprot_id
            ,source_id
    """).to_df()

    return pubmed_ids

    # stream = Entrez.einfo(db="pubmed")
    # record = Entrez.read(stream)
    # for field in record["DbInfo"]["FieldList"]:
    #     print("%(Name)s, %(FullName)s, %(Description)s" % field)
