from typing import Generator

import duckdb
import pandas as pd


def generate(data_path) -> Generator:
    df = pd.read_parquet(data_path)

    docs = duckdb.sql("""
        SELECT
            uniprot_id
            ,string_agg(distinct value, '\n') FILTER (WHERE field = 'comment' AND type not in ('SUBUNIT', 'PATHWAY', 'FUNCTION') ) as info
            ,string_agg(distinct value, '\n') FILTER (WHERE field = 'name') as names
            ,COALESCE(string_agg(distinct value, '\n') FILTER (WHERE field = 'comment' AND type = 'SUBUNIT'), 'No known interactions') as interactions
            ,COALESCE(string_agg(distinct value, '\n') FILTER (WHERE field = 'comment' AND type = 'PATHWAY'), 'No known pathways') as pathways
            ,COALESCE(string_agg(distinct value, '\n') FILTER (WHERE field = 'comment' AND type = 'FUNCTION'), 'No known functions') as functions
        FROM df
        GROUP BY uniprot_id
    """).to_df()
    for row in docs.itertuples():
        contents = f"""# Uniprot ID: "{row.uniprot_id}"

## {row.uniprot_id} Info
{row.info}

## {row.uniprot_id} Alternative Names
{row.names}

## {row.uniprot_id} Interactions
{row.interactions}

## {row.uniprot_id} Functions
{row.functions}

## {row.uniprot_id} Pathways
{row.pathways}
"""
        yield f"{row.uniprot_id.lower()}.txt", contents
