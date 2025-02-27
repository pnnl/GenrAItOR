import datetime

def postprocess_uniprot(
        results,
        uniprots
    ):

    contexts = []

    for abstracts, path_info, path_text, interaction_text, uniprot in zip(
        results['abstracts'],
        results['xref_reactome'],
        results['cc_pathway'],
        results['cc_subunit'],
        uniprots
    ):
        abstracts = [a for a in abstracts if a is not None]
        cur_abstracts = f"ABSTRACTS ({uniprot}):\n" + "\n".join(abstracts) + "\n" if len(abstracts) > 0 else ""

        cur_interactions = f"INTERACTIONS ({uniprot}):\n" + "\n".join(interaction_text) + "\n" if len(interaction_text) > 0 else ""

        if len(path_info) > 0 or len(path_text) > 0:
            cur_pathways = f"PATHWAY INFO ({uniprot}):\n"

        if len(path_info) > 0:
            append_pathways = []
            for entry in path_info:
                plus_context = entry['id'] + ": " + ".  ".join([el['value'] for el in entry['properties']])
                append_pathways.append(plus_context)
            cur_pathways += "\n".join(append_pathways) + "\n"

        if len(path_text) > 0:
            cur_pathways = cur_pathways + "\n".join(path_text) + "\n" if len(path_text) > 0 else ""

        contexts.append(cur_abstracts + cur_interactions + cur_pathways)
        
    contexts = [f"{uniprot}\n" + c for uniprot,c in zip(uniprots, contexts)]

    full_context = "### Begin Context For: " + "### End Context\n\n### Begin Context For: ".join(contexts) + "### End Context"

    thetime = datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")
    with open(f"context_{len(uniprots)}_uniprots_{thetime}.txt", "w") as f:
        f.write(full_context)

    return full_context