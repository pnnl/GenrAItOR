"""Evaluation commands."""

from pathlib import Path

import click

from ..conf import env, log

@click.group()
def cli():
    """Grouping eval cli commands."""
    pass


@cli.command("eval:init")
def init():
    """Install nltk dependencies."""
    import ssl

    import nltk

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download("punkt_tab")


@cli.command("eval")
@click.option(
    "-a",
    "--adapter_path",
    "adapter_path",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-b",
    "--base_model",
    "--base",
    "base_model",
    default=env.model.name,
    show_default=True,
)
@click.option(
    "--raft_path",
    required=True,
    default=Path(env.paths.data) / "training" / "raft_outputs" / "raw.jsonl",
    type=click.Path(exists=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--save_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--batch_size", "batch_size", type=int, default=15)
def evaluate(adapter_path, base_model, raft_path, save_path, batch_size):
    """Run the AlignScore metric."""
    import pandas as pd

    from ..evaluate import align
    from ..raft import train

    data = align.load(raft_path)
    tokenizer, model = train.load(
        base_model=base_model,
        adapter_path=None,
    )
    base_result = align.evaluate(
        model=model, tokenizer=tokenizer, data=data, batch_size=batch_size
    )
    base_result["model"] = "base"

    tokenizer, model = train.load(
        base_model=base_model,
        adapter_path=adapter_path,
    )
    tuned_result = align.evaluate(
        model=model, tokenizer=tokenizer, data=data, batch_size=batch_size
    )
    tuned_result["model"] = "tuned"
    result = pd.concat([tuned_result, base_result])
    print(result.describe())
    if save_path is None:
        print(result.to_markdown(index=False))
    else:
        result.to_parquet(save_path, index=False)
