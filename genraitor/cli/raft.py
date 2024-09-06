"""."""

from pathlib import Path

import click

from ..conf import env, log


@click.group()
def cli():
    """."""
    pass


@cli.command("raft:merge")
@click.option(
    "-a",
    "--adapter_path",
    "adapter_path",
    required=True,
    type=click.Path(file_okay=False, path_type=Path, exists=True),
)
@click.option(
    "-b",
    "--base_model",
    "--base",
    "base_model",
    default=env.model.name,
)
@click.option("-o", "--output_path", "--save_path", "save_path", default="genraitor")
def merge(adapter_path, base_model, save_path):
    """Merge the trained model with the base model."""
    from ..raft import train

    model = train.load(
        adapter_path=adapter_path,
        base_model=base_model,
    )
    log.info("saving merged")
    model.save_pretrained(save_path)
    log.info(f"saved: {save_path}")


@cli.command("raft:data")
@click.option(
    "-o",
    "--output_path",
    "save_path",
    required=True,
    default=Path(env.paths.data) / "training" / "raft_outputs",
    type=click.Path(file_okay=False, path_type=Path),
)
def dataset_raft(save_path):
    """Generate raft training dataset from uniprot documents."""
    from llama_index.embeddings.ollama import OllamaEmbedding
    from llama_index.llms.ollama import Ollama

    from ..data.raft_dataset import RAFTDatasetPack

    log.info(f"loading: {env.paths.rag_data}")
    llm = Ollama(model="llama3.1", request_timeout=120)
    embed_model = OllamaEmbedding(
        model_name="llama3.1",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={"mirostat": 0, "request_timeout": 120},
    )
    raft_dataset = RAFTDatasetPack(env.paths.rag_data, llm=llm, embed_model=embed_model)
    dataset = raft_dataset.run()

    assert save_path.parent.exists(), f"{save_path} parent not found"
    # Save as .arrow format
    dataset.save_to_disk(save_path)

    # Save as .jsonl format
    dataset.to_json((save_path / "raw").with_suffix(".jsonl"))
