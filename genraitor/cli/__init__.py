import click

from ..tune import main


@click.group()
def cli():
    pass


@cli.command()
def tune():
    main()
