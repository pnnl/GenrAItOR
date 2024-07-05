"""CLI commands for data generation."""
from pathlib import Path
from pprint import pprint

import click
import pandas as pd

from ..conf import env, log


@click.group()
def cli():
    """dataset:group."""
    pass


@cli.command("data:uniprot")
@click.option(
    "-o",
    "--output_path",
    "save_path",
    required=True,
    default=Path(env.paths.data) / "training" / "uniprot.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def dataset_uniprot_from_deepimv(save_path: Path):
    """Generate uniprot ids from deepimv data"""
    from ..data import deepimv, uniprot

    data_path = Path(env.paths.data) / "deepimv"
    assert data_path.exists(), f"shapley files not found {data_path}"
    proteins = deepimv.top_shapley(data_path)

    data = uniprot.main(proteins)
    match save_path:
        case None:
            pprint(data.to_markdown(index=False))
        case _ if save_path.suffix == ".parquet":
            data.to_parquet(save_path, index=False)
            log.info(f"saved: {save_path}")
        case _ if save_path.suffix == ".csv":
            data.to_csv(save_path, index=False)
            log.info(f"saved: {save_path}")
        case _:
            log.warning(f"{save_path} not supported yet")


@cli.command("data:uniprot-to-pubmed")
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
    required=False,
    # default=Path(env.paths.data) / "training" / "uniprot_pubmed_ids.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def dataset_uniprot_to_pubmed_ids(uniprot_path, save_path):
    """Generate pubmed ids from uniprot data."""
    from ..conf import log
    from ..data import pubmed

    df = pd.read_parquet(uniprot_path)
    ids = pubmed.ids_from_uniprot(df)
    match save_path:
        case None:
            pprint(ids.to_markdown(index=False))
        case _ if not save_path.parent.exists():
            raise click.UsageError(f"dir does not exist: {save_path.parent}")
        case _ if save_path.suffix == ".parquet":
            ids.to_parquet(save_path, index=False)
            log.info(f"saved: {save_path}")
        case _ if save_path.suffix == ".csv":
            ids.to_csv(save_path, index=False)
            log.info(f"saved: {save_path}")


@cli.command("data:pubmed")
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
    required=False,
    default=Path(env.paths.data) / "training" / "pubmed_articles.parquet",
    type=click.Path(dir_okay=False, path_type=Path),
)
def dataset_pubmed(uniprot_path, save_path):
    """Generate pubmed abstracts from uniprot data."""
    from ..data import pubmed

    df = pd.read_parquet(uniprot_path)
    ids = pubmed.ids_from_uniprot(df)

    articles = pubmed.articles(ids)
    articles.to_parquet(save_path, index=False)
    log.info(f"saved: {save_path}")
