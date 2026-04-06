from click.testing import CliRunner
from agentune.cli import cli


class TestBaselineBackendSupport:
    def test_baseline_accepts_backend_option(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["baseline", "--help"])
        assert "--backend" in result.output
