import click

from .data import cli as data_cli
from .evaluate import cli as evaluate_cli
from .raft import cli as raft_cli
from .rag import cli as rag_cli
from .train import cli as train_cli

main_cli = click.CommandCollection(
    help="Genraitor CLI application",
    sources=[data_cli, raft_cli, rag_cli, evaluate_cli, train_cli],
)
