from metapub import PubMedFetcher
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

DEFAULT_QUERY = "reviewed:true+AND+{uniprot}&fields=lit_pubmed_id,cc_interaction,cc_subunit,gene_synonym"
DEFAULT_HEADERS = {
    'Accept': 'application/json',
    'Content-Type': 'application/json'
}

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

def extract_pathways(query_results, databases = ['Reactome']):
    """Extract the pathway information from the results of a uniprot query.

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

        url = f"https://rest.uniprot.org/uniprotkb/search?query={DEFAULT_QUERY.format(uniprot=uniprot)}"

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

def fetch_context(uniprots, secondary_context = False, max_characters=16_000, **kwargs):
    """Fetch context information for a list of uniprot accession numbers.

    Args:
        uniprots (List[str]): A list of uniprot accession numbers to query.
        secondary_context (bool, optional): Whether to fetch results of secondary search results. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the fetch_uniprot function.

    Returns:
        Tuple[List[str], List[str]]: Context information for each uniprot query as a string suitable for injection into a prompt template.  The first list contains abstracts, the second list contains interaction information.
    """
    # A list of results per uniprot query.  Each result is a list of dictionaries
    query_results = fetch_uniprot(uniprots, **kwargs)

    all_abstract_context = []
    all_interaction_context = []
    all_pathway_context = []

    # For each query result
    for up, qr in zip(uniprots, query_results):
        abstract_context = "ABSTRACTS:\n\n<DOCUMENT>\n"
        interaction_context = "INTERACTIONS:\n\n"
        pathway_context = "PATHWAYS:\n\n"

        # only keep information from the direct hit of the query
        if not secondary_context:
            qr = [el for el in qr if el['primaryAccession'] == up]

        abstracts, _ = fetch_abstracts(qr)
        interactions = extract_interactions(qr)
        pathways = extract_pathways(qr)

        for a in abstracts:
            if a is None:
                continue
            if (len(abstract_context) + len(a)) > max_characters:
                break
            abstract_context += a + "<\DOCUMENT>\n\n<DOCUMENT>\n"

        # strip the trailing <DOCUMENT> tag
        abstract_context = abstract_context[:-len("<DOCUMENT>\n")]

        for i in interactions:
            interaction_context += json.dumps(i) + "\n\n"

        for p in pathways:
            pathway_context += json.dumps(p) + "\n\n"

        all_abstract_context.append(abstract_context)
        all_interaction_context.append(interaction_context)
        all_pathway_context.append(pathway_context)

    return all_abstract_context, all_interaction_context, all_pathway_context