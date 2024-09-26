from metapub import PubMedFetcher
import requests
import logging

logging.basicConfig(level=logging.INFO)

def extract_interactions(query_results):
    """Extract the interaction information from the results of a uniprot query.

    Args:
        query_results (List): A list of dictionaries containing the results of a single uniprot query from `fetch_uniprot`.

    Returns:
        List: A list of dictionaries containing all interaction information from all query results.
    """
    all_interactions = []

    for qr in query_results:
        if 'comments' in qr:
            for c in qr['comments']:
                if 'interactions' in c:
                    for i in c['interactions']:
                        all_interactions.append(i)
    
    return all_interactions

def extract_pathways_comments(query_results):
    """Extract comments about pathway information obtained through the cc_pathways field in the uniprot api.
        Args:
            query_results (List): A list of dictionaries containing the results of a single uniprot query from `fetch_uniprot`.
        Returns:
            List: A list of dictionaries containing all pathway information from all query results.
    """
    all_comments = []

    for qr in query_results:
        if 'comments' in qr:
            for c in qr['comments']:
                if c['commentType'] == 'PATHWAY':
                    for txt in c['texts']:
                        all_comments.append(txt['value'])

    return(all_comments)

def extract_pathways_db(query_results, databases = ['Reactome']):
    """Extract the pathway information from the results of a uniprot query.  The appropriate field must have been supplied to the uniprot query, e.g. 'xref_reactome' for the Reactome database.  See https://www.uniprot.org/help/return_fields_databases

    Args:
        query_results (List): A list of dictionaries containing the results of a single uniprot query from `fetch_uniprot`.
        databases (List[str]): A list of database names returned by the uniprot api from which to retrieve pathway information from.
    Returns:
        List: A list of dictionaries containing all pathway information from all query results.
    """
    all_pathways = []

    for qr in query_results:
        if 'uniProtKBCrossReferences' in qr:
            for xref in qr['uniProtKBCrossReferences']:
                if xref['database'] in databases:
                    tmp_xref = xref.copy()
                    tmp_xref['properties'] = [p for p in tmp_xref['properties'] if len(p['value']) > 1]
                    tmp_xref = {k:v for k,v in tmp_xref.items() if k != 'database'}
                    all_pathways.append(tmp_xref)
    
    return all_pathways

def extract_subunit(query_results):
    """
    Args:
        query_results (List): A list of dictionaries containing the results of a single uniprot query from `fetch_uniprot`.
    Returns:
        List: A list of texts describing interactions for each query result.
    """
    all_comments = []

    for qr in query_results:
        if 'comments' in qr:
            for c in qr['comments']:
                if c['commentType'] == 'SUBUNIT':
                    for txt in c['texts']:
                        all_comments.append(txt['value'])

    return all_comments
                    
QUERY_BASE = "reviewed:true+AND+{uniprot}&fields=lit_pubmed_id{extra_fields}"
DEFAULT_HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

EXTRACT_FN_MAP = {
    "cc_interaction":  extract_interactions,
    "cc_pathway": extract_pathways_comments,
    "xref_reactome": extract_pathways_db,
    "cc_subunit": extract_subunit
}

DEFAULT_EXTRA_FIELDS = list(EXTRACT_FN_MAP.keys())
DEFAULT_QUERY = QUERY_BASE.format(uniprot = "{uniprot}", extra_fields = "," + ",".join(DEFAULT_EXTRA_FIELDS))

def fetch_uniprot(uniprots, headers = DEFAULT_HEADERS, payload = {}, query_body=DEFAULT_QUERY):
    """Fetch uniprot information for a list of uniprot ids using the uniprot REST API.

    Args:
        uniprots (List[str]): A list of uniprot accession numbers to query.  Other query forms (like uniprot id) will still return results, but may be too large to handle.
        headers (dict): Headers to pass to the request. Defaults to DEFAULT_HEADERS.
        payload (dict, optional): Data payload to pass to the request. Defaults to {}.
        query_body (str, optional): Query string following the `query` header parameter of the uniprot API query. Defaults to DEFAULT_QUERY.

    Returns:
        List[List[dict]]: A list of results per uniprot query.  Each result is a list of dictionaries with results returned from searching for that uniprot accession number.
    """
    all_results = []

    for uniprot in uniprots:

        url = f"https://rest.uniprot.org/uniprotkb/search?query={query_body.format(uniprot=uniprot)}"

        r = requests.request("GET", url, headers=headers, data=payload)
        
        json_response = r.json()
        
        all_results.append(json_response['results'])

    return all_results

def fetch_abstracts(query_results, cache_dir = "~/.cache/metapub"):
    """Fetch abstracts for a list of query results from the uniprot API.

    Args:
        query_results (List): A list of dictionaries containing the results of a single uniprot query from `fetch_uniprot`.
        cache_dir (str, optional): Directory to cache pubmed articles. Defaults to "~/.cache/metapub".
    """

    fetcher = PubMedFetcher(cachedir = cache_dir)

    all_abstracts = []
    all_pmids = []

    for record in query_results:
        if 'references' in record:
            for ref in record['references']:
                citation = ref['citation']
                if 'citationCrossReferences' in citation:
                    for c in citation['citationCrossReferences']:
                        if c.get('database') == 'PubMed':
                            pmid = c.get('id')
                            if pmid in all_pmids:
                                continue
                            article = fetcher.article_by_pmid(pmid)
                            all_abstracts.append(article.abstract)
                            all_pmids.append(pmid)
    
    return all_abstracts, all_pmids

def fetch_context(uniprots, secondary_context = False, max_characters=16_000, extra_fields=list(EXTRACT_FN_MAP.keys()), **kwargs):
    """Fetch context information for a list of uniprot accession numbers.

    Args:
        uniprots (List[str]): A list of uniprot accession numbers to query.
        secondary_context (bool, optional): Whether to fetch results of secondary search results. Defaults to False.
        max_characters (int, optional):  Limit on the number of characters in the abstract information to return.  Defaults to 16_000.
        extra_fields (List[str], optional):  A list of strings for the comma separated entries following the `fields` header parameter in the uniprot query.  See https://www.uniprot.org/help/api_queries for examples.
        **kwargs: Additional keyword arguments to pass to the fetch_uniprot function.

    Returns:
        Dict[str, List[Any]]: Each value of the returned dictionary contains a list with each element corresponding to the entry in the uniprots argument.  They are abstracts as well as information from the `extra_fields` argument passed to the uniprot query.
    """
    uniprot_query = DEFAULT_QUERY.format(uniprot = "{uniprot}", extra_fields = "," + ",".join(extra_fields))

    # A list of results per uniprot query.  Each result is a list of dictionaries
    query_results = fetch_uniprot(uniprots, query_body = uniprot_query, **kwargs)

    all_context = {}

    # For each query result
    for up, qr in zip(uniprots, query_results):
        logging.info(f"Retrieving context for uniprot id {up}")
        # abstract_context = []
        # interaction_context = []
        # pathway_db_context = []

        # only keep information from the direct hit of the query
        if not secondary_context:
            qr = [el for el in qr if el['primaryAccession'] == up]

        # always fetch abstract context
        abstracts, _ = fetch_abstracts(qr)

        # fetch other context information
        for field in extra_fields:
            if field in EXTRACT_FN_MAP:
                res = EXTRACT_FN_MAP[field](qr)
                all_context.setdefault(field, []).append(res)
            else:
                logging.warning(f"Field {field} not found in extraction function map.")

        abstract_len = 0

        # pointless, it's already a list...we're just re-appending to another list...just check the length.
        for i, a in enumerate(abstracts):
            if a is None:
                continue
            if (abstract_len + len(a)) > max_characters:
                break
            abstract_len += len(a)
        
        all_context.setdefault('abstracts', []).append(abstracts[:i])

    return all_context
