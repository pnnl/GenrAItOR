"""Training CLI."""

from pathlib import Path

import click

from ..conf import env
from ..raft.strategies import TrainingStrategy


@click.group()
def cli():
    """Training commands."""
    pass


@cli.command("train:raft")
@click.option(
    "-t",
    "--training_path",
    "training_path",
    required=True,
    default=Path(env.paths.data) / "training_dataset.jsonl",
    type=click.Path(path_type=Path, exists=True),
    help="path to the training dataset (the RAFT llamapack outputs).",
    show_default=True,
)
@click.option(
    "-m",
    "--model_name",
    "--model",
    default="meta-llama/Meta-Llama-3-8B",
    help="HF model to use as the base, either a name referencing a huggingface model, or a path to a local huggingface model folder",
    show_default=True,
)
@click.option(
    "-n",
    "--output_name",
    default="data/finetuned",
    help="Output path for the adapter weights.",
    show_default=True,
)
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
    from ..raft import train

    strategy = TrainingStrategy.from_str(strategy)
    train.main(
        training_path=training_path,
        base_model=model_name,
        new_model=output_name,
        strategy=strategy,
    )
