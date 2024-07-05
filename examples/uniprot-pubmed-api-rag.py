from metapub import PubMedFetcher
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

import requests
import json
import argparse
import logging
import os

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

def fetch_context(uniprots, secondary_context = False, **kwargs):
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

    # For each query result
    for up, qr in zip(uniprots, query_results):
        abstract_context = "ABSTRACTS:\n\n"
        interaction_context = "INTERACTIONS:\n\n"

        # only keep information from the direct hit of the query
        if not secondary_context:
            qr = [el for el in qr if el['primaryAccession'] == up]

        abstracts, _ = fetch_abstracts(qr)
        interactions = extract_interactions(qr)

        for a in abstracts:
            abstract_context += a + "\n\n"

        for i in interactions:
            interaction_context += json.dumps(i) + "\n\n"

        all_abstract_context.append(abstract_context)
        all_interaction_context.append(interaction_context)

    return all_abstract_context, all_interaction_context

def main():
    parser = argparse.ArgumentParser(description="Query the Uniprot API and pubmed api for context to ask questions about proteins")
    parser.add_argument("--uniprots", nargs="+", help="Uniprot accession numbers to query")
    parser.add_argument("--secondary-context", action="store_true", help="Include all context information, not just the direct hit of the query")
    parser.add_argument("--cache-dir", default="~/.cache/metapub", help="Directory to cache pubmed articles")
    parser.add_argument("--api-key-path", help="Path to the file containing the OpenAI API key")
    parser.add_argument("--api-base", default="https://ai-incubator-api.pnnl.gov", help="Base URL for the OpenAI API")
    parser.add_argument("--model", default="gpt-4o", help="Model to use for the OpenAI API")
    parser.add_argument("--template-file", help="Path to the file containing the template for the query prompt, must contain format options for context_str and query_str")
    parser.add_argument("--query-file", help="Path to file containing the query string")
    args = parser.parse_args()

    if args.api_key_path:
        with open(args.api_key_path) as f:
            API_KEY=f.readline()
    else:
        # should work automatically but meh.
        API_KEY=os.environ['OPENAI_API_KEY']

    llm = OpenAI(
        model=args.model,
        api_key=API_KEY,
        api_base=args.api_base
    )

    ## Simple RAG
    if args.template_file:
        with open(args.template_file) as f:
            template = f.read()

        qa_prompt = PromptTemplate(template)
    else:
        logging.info("No template file provided, using default template")
        qa_prompt = PromptTemplate(
            """\
        Context information is below.
        ---------------------
        {context_str}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query_str}
        Step-by-step answer: \
        """
        )

    if args.query_file:
        with open(args.query_file) as f:
            query_str = f.read()
    else:
        logging.info("No query file provided, using default query")
        # example
        query_str = "Does P05937 interact directly or indirectly with Q9NQA5?  Provide details about the nature of the interactions from the abstracts if possible."

    abstracts, interactions = fetch_context(args.uniprots)
    contexts = [a + i for a, i in zip(abstracts, interactions)]

    all_fmt_qa_prompts = [qa_prompt.format(context_str=c, query_str=query_str) for c in contexts]

    all_responses = []

    for fmt_qa_prompt in all_fmt_qa_prompts:
        response = llm.complete(fmt_qa_prompt)
        all_responses.append(str(response))

    print("\n\n".join(all_responses))

if __name__ == "__main__":
    main()