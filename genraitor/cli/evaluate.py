"""Evaluation commands."""

from pathlib import Path

import click
import duckdb
import pandas as pd

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
    from alignscore import AlignScore
    with duckdb.connect(":memory:") as conn:
        data = conn.sql(f"""
            SELECT
                context || instruction || question as context
                ,cot_answer as claim
            FROM read_json("{raft_path}")
        """).to_df()
    claims = data["claim"].to_list()[:batch_size]
    contexts = data["context"].to_list()[:batch_size]


    tokenizer, model = train.load(
        adapter_path=adapter_path,
        base_model=base_model,
    )
    model.eval()

    device = model.device
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(contexts, return_tensors="pt", padding=True)
    ids = inputs["input_ids"].to(device)

    outputs = model.generate(input_ids=ids, max_new_tokens=150, pad_token_id=tokenizer.pad_token_id)
    pred_claims = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)


    scorer = AlignScore(
        model="roberta-base",
        batch_size=32,
        device=device,
        ckpt_path=str(Path(env.paths.app) / "data/alignscore/AlignScore-base.ckpt"),
        evaluation_mode="nli_sp",
    )
    scores = scorer.score(contexts=contexts, claims=pred_claims)
    pred_scores = pd.DataFrame(scores, columns=["align_score"])
    pred_scores["dataset"] = "pred"

    scores = scorer.score(contexts=contexts, claims=claims)
    eval_scores = pd.DataFrame(scores, columns=["align_score"])
    eval_scores["dataset"] = "eval"

    result = pd.concat([eval_scores, pred_scores])

    print(result.describe())
    if save_path is None:
        print(result.to_markdown(index=False))
    else:
        result.to_parquet(save_path, index=False)
