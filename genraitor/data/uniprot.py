import pandas as pd
from ..conf import log

# import duckdb
import requests
import numpy as np


def main(proteins) -> pd.DataFrame:
    results = []
    for protein in proteins["name"]:
        url = f"https://rest.uniprot.org/uniprotkb/search?query={protein}&format=json"
        log.info(f"harvesting {protein}: {url}")
        response = requests.get(url)
        out = extract(protein, response)
        results.append(out)
    return pd.concat(results)


def extract(uniprot_id, response) -> pd.DataFrame:
    content = response.json()
    if len(content["results"]) > 1:
        log.warning(f"found {len(content['results'])} results for {uniprot_id}")
        result = [d for d in content["results"] if d["uniProtkbId"] == uniprot_id][0]
    else:
        result = content["results"][0]
    comments = extract_comments(result["comments"])
    names = extract_names(result["proteinDescription"])
    c = pd.DataFrame(comments)
    n = pd.DataFrame(names)
    df = pd.concat([c, n]).fillna(value=np.nan)
    df["uniprot_id"] = uniprot_id
    return df


def extract_names(description):
    results = []
    out = {}
    out["field"] = "name"
    out["type"] = "recommended"
    out["value"] = (
        description.get("recommendedName", None)
        .get("fullName", None)
        .get("value", None)
    )
    results.append(out)
    for name in description.get("alternativeNames", []):
        out = {}
        out["field"] = "name"
        out["type"] = "alias"
        out["value"] = name.get("fullName", None).get("value", None)
        results.append(out)
    return results


def extract_comments(comments):
    results = []
    for comment in comments:
        out = {}
        out["field"] = "comment"
        out["type"] = comment["commentType"].lower()
        for text in comment.get("texts", []):
            out["value"] = text["value"]
            for evidence in text.get("evidences", []):
                out["source_name"] = evidence.get("source", None)
                out["source_id"] = evidence.get("id", None)
                results.append(out.copy())
            if "evidences" not in text:
                results.append(out.copy())
    return results
