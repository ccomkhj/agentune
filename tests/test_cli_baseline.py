from click.testing import CliRunner
from agentune.cli import cli


class TestBaselineBackendSupport:
    def test_baseline_accepts_backend_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["baseline", "--help"])
        assert "--backend" in result.output


class TestCompareCommand:
    def test_compare_help_shows_options(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "--dataset" in result.output
        assert "--total-trials" in result.output
        assert "--backend" in result.output
        assert "--trials-per-round" in result.output
