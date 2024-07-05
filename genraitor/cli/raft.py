"""."""
from pathlib import Path

import click

from ..conf import env, log
from ..raft.strategies import TrainingStrategy


@click.group()
def cli():
    """."""
    pass


@cli.command("raft:tune")
@click.option(
    "-t",
    "--training_path",
    "training_path",
    required=True,
    default=Path(env.paths.data) / "training_dataset.jsonl",
    type=click.Path(dir_okay=False, path_type=Path, exists=True),
)
@click.option(
    "-m", "--model_name", "--model", default="meta-llama/Meta-Llama-3-8B", help="HF model to use as the base",
)
@click.option("-n", "--output_name", default="data/finetuned", help="Output path for the adapter weights.")
@click.option(
    "-",
    "--strategy",
    default="sft",
    type=click.Choice(TrainingStrategy.list()),
    show_choices=True,
    show_default=True,
    help="TRL trainer to use",
)
def tune(training_path, model_name, output_name, strategy):
    """Tune llm using generated raft dataset."""
    from ..raft import tune

    strategy = TrainingStrategy.from_str(strategy)
    tune.main(
        training_path=training_path,
        base_model=model_name,
        new_model=output_name,
        strategy=strategy,
    )


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
    from ..raft import tune

    model = tune.load(
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
