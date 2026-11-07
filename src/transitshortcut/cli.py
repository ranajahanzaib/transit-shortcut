from __future__ import annotations

import json
from typing import Annotated

import typer

from .data import load, prepare
from .experiments import get_run, markdown, runs
from .experiments import train as run_train

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
data = typer.Typer(no_args_is_help=True)
app.add_typer(data, name="data")


@data.command("prepare")
def prepare_data(
    mode: str = "sample",
    seed: int = 42,
    train_size: int = 2400,
    validation_size: int = 500,
    test_size: int = 500,
    revision: str = "main",
    max_flux_mb: int = 512,
) -> None:
    """Prepare a revision-pinned snapshot, enforcing a compressed flux budget."""
    result = prepare(
        mode,
        seed,
        {"train": train_size, "validation": validation_size, "test": test_size},
        revision,
        max_flux_mb,
    )
    typer.echo(json.dumps(result["summary"], indent=2))


@data.command()
def inspect() -> None:
    typer.echo(json.dumps(load()[1], indent=2))


@app.command()
def train(experiment: str = "multiclass") -> None:
    result = run_train(experiment)
    typer.echo(result["id"])
    for name, model in result["models"].items():
        typer.echo(f"{name}: {json.dumps(model['test'])}")


def resolve(experiment: str) -> dict:
    choices = [r for r in runs() if r["experiment"] == experiment or r["id"] == experiment]
    if not choices:
        raise typer.BadParameter("No matching completed experiment. Train it first.")
    return get_run(choices[0]["id"])


@app.command()
def audit(experiment: str = "multiclass") -> None:
    """Print the refit audit computed and persisted during training."""
    typer.echo(json.dumps(resolve(experiment)["ablations"], indent=2))


@app.command()
def report(
    experiment: str = "multiclass",
    format: str = "markdown",
    output: Annotated[str | None, typer.Option()] = None,
) -> None:
    from pathlib import Path

    from .experiments import read_predictions

    result = resolve(experiment)
    if format == "markdown":
        content = markdown(result)
    elif format == "json":
        content = json.dumps(result, indent=2)
    elif format == "csv":
        content = read_predictions(result).to_csv(index=False)
    else:
        raise typer.BadParameter("Use markdown, json, or csv")
    if output:
        Path(output).write_text(content)
    else:
        typer.echo(content)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("transitshortcut.api:app", host=host, port=port)


def main() -> None:
    try:
        app()
    except (ValueError, FileNotFoundError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from None
