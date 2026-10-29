import typer

from rek.commands.build import app as build_app
from rek.commands.eval import app as eval_app
from rek.commands.inspect import app as inspect_app
from rek.commands.kb import app as kb_app
from rek.commands.train import app as train_app


app = typer.Typer(
    help="Repo Evolution Knowledge CLI",
    no_args_is_help=True,
)

app.add_typer(build_app, name="build")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(kb_app, name="kb")
app.add_typer(inspect_app, name="inspect")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
