import click
from pathlib import Path
from ..conf import log, env
from ..models.embedding import EmbedModel


@click.group()
def cli():
    pass


def response_modes():
    from llama_index.core.response_synthesizers.type import ResponseMode

    return [m.value for m in ResponseMode]


@cli.command("rag:run")
@click.option("-k", "--top_k", "top_k", default=5, type=int)
@click.option(
    "-r",
    "--response_mode",
    default="tree_summarize",
    type=click.Choice(response_modes()),
)
@click.option(
    "-e",
    "--embedding_model",
    default=EmbedModel.LLAMA3_1,
    type=click.Choice([m.value for m in EmbedModel]),
)
@click.option("--reindex", default=False, is_flag=True)
def rag_run(top_k, response_mode, embedding_model, reindex):
    """run rag model"""
    from ..rag import run

    run.main(
        top_k=top_k,
        response_mode=response_mode,
        embedding_model=embedding_model,
        reindex=reindex,
    )


@cli.command("rag:index")
@click.argument("prompt", default="lacrt_human")
@click.option("-k", "--top_k", "top_k", default=5, type=int)
def rag_index(prompt, top_k):
    """generate rag index and embeddings from uniprot documents"""
    from ..rag import embedding

    from ..conf import llm_model, embed_model

    index = embedding.get_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(prompt)
    for node in nodes:
        log.info(node.metadata["file_name"])
        log.info(node.score)


@cli.command("rag:data")
@click.option(
    "-f",
    "--uniprot_path",
    "uniprot_path",
    required=False,
    default=Path(env.paths.data) / "training" / "uniprot.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "-o",
    "--output_path",
    "save_path",
    required=True,
    default=Path(env.paths.rag_data),
    type=click.Path(file_okay=False, path_type=Path),
)
def dataset_rag(uniprot_path, save_path):
    """generate generate uniprot documents"""
    from ..rag import data

    log.info("generating rag documents from uniprot data")
    docs = data.generate(uniprot_path)
    for filename, doc in docs:
        with open(save_path / filename, "w") as f:
            f.write(doc)
        log.info(f"saved: {save_path / filename}")
