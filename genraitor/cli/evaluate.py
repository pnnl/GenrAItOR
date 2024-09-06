"""Evaluation commands."""

from pathlib import Path

import click

from ..conf import env
from ..raft import train


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
@click.option(
    "--raft_path",
    required=True,
    default=Path(env.paths.data) / "training" / "raft_outputs" / "raw.jsonl",
    type=click.Path(dir_okay=False, exists=True, path_type=Path),
)
@click.option(
    "--save_path",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--batch_size", "batch_size", type=int, default=15)
def evaluate(adapter_path, base_model, raft_path, save_path, batch_size):
    """Run the AlignScore metric."""
    import duckdb

    from ..evaluate import align

    with duckdb.connect(":memory:") as conn:
        data = conn.sql(f"""
            SELECT
                context || instruction || question as context
                ,cot_answer as claim
            FROM read_json("{raft_path}")
        """).to_df()

    tokenizer, model = train.load(
        adapter_path=adapter_path,
        base_model=base_model,
    )
    model.eval()

    result = align.evaluate(model=model, tokenizer=tokenizer, data=data, batch_size=batch_size)
    print(result.describe())
    if save_path is None:
        print(result.to_markdown(index=False))
    else:
        result.to_parquet(save_path, index=False)
