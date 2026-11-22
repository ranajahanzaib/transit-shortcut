import json

from typer.testing import CliRunner

from transitshortcut.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout
    assert "audit" in result.stdout
    assert "report" in result.stdout
    assert "serve" in result.stdout
    assert "data" in result.stdout


def test_cli_data_help():
    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "prepare" in result.stdout
    assert "inspect" in result.stdout


def test_cli_inspect(cohort):
    result = runner.invoke(app, ["data", "inspect"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["dataset_id"] == "synthetic-test"
    assert "summary" in data


def test_cli_train_and_audit(cohort):
    train_res = runner.invoke(app, ["train", "--experiment", "multiclass"])
    assert train_res.exit_code == 0
    assert "multiclass-" in train_res.stdout
    assert "shortcut:" in train_res.stdout
    assert "signal:" in train_res.stdout

    audit_res = runner.invoke(app, ["audit", "--experiment", "multiclass"])
    assert audit_res.exit_code == 0
    ablations = json.loads(audit_res.stdout)
    assert isinstance(ablations, list)
    assert len(ablations) == 4
    assert any(a["group"] == "Transit availability" for a in ablations)


def test_cli_report(cohort, tmp_path):
    runner.invoke(app, ["train", "--experiment", "multiclass"])

    # JSON report
    json_res = runner.invoke(app, ["report", "--experiment", "multiclass", "--format", "json"])
    assert json_res.exit_code == 0
    data = json.loads(json_res.stdout)
    assert data["experiment"] == "multiclass"

    # Markdown report
    md_res = runner.invoke(app, ["report", "--experiment", "multiclass", "--format", "markdown"])
    assert md_res.exit_code == 0
    assert "# TransitShortcut audit:" in md_res.stdout

    # CSV report
    csv_res = runner.invoke(app, ["report", "--experiment", "multiclass", "--format", "csv"])
    assert csv_res.exit_code == 0
    assert "target_id,star_id,mission,disposition" in csv_res.stdout

    # Output file writing
    out_file = tmp_path / "test_report.md"
    file_res = runner.invoke(
        app,
        ["report", "--experiment", "multiclass", "--format", "markdown", "--output", str(out_file)],
    )
    assert file_res.exit_code == 0
    assert out_file.exists()
    assert "# TransitShortcut audit:" in out_file.read_text()


def test_cli_errors(cohort):
    bad_exp = runner.invoke(app, ["report", "--experiment", "nonexistent"])
    assert bad_exp.exit_code != 0

    bad_fmt = runner.invoke(app, ["report", "--experiment", "multiclass", "--format", "xml"])
    assert bad_fmt.exit_code != 0
