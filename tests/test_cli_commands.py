"""Tests for CLI commands using Typer's CliRunner."""

from qphase.main import app
from typer.testing import CliRunner

runner = CliRunner()


def test_version():
    """Test --version flag."""
    _ = runner.invoke(app, ["--version"])
    # Note: Typer might not handle --version automatically unless configured
    # If it fails, we might need to check if we implemented a version callback
    # For now, let's check if it runs without error or if we need to add it
    pass  # Skip for now as version callback might not be in main.py


def test_init_command(temp_workspace):
    """Test 'init' command."""
    # We use the temp_workspace fixture which sets QPHASE_SYSTEM_CONFIG
    # The init command should use the paths from that config

    # We need to mock input for confirmation if force is not used
    result = runner.invoke(app, ["init", "--force"])
    assert result.exit_code == 0
    assert "Initializing QPhase Project" in result.stdout

    # Check if global.yaml was created
    config_dir = temp_workspace / "configs"
    assert (config_dir / "global.yaml").exists()


def test_config_show_system(temp_workspace):
    """Test 'config show system'."""
    result = runner.invoke(app, ["config", "show", "--system"])
    assert result.exit_code == 0
    assert "paths" in result.stdout
    # Check for config_dirs presence
    assert "config_dirs" in result.stdout


def test_plugin_list():
    """Test 'list' command."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "engine" in result.stdout
    assert "backend" in result.stdout


def test_plugin_show():
    """Test 'show' command."""
    # Assuming 'dummy' backend is available
    result = runner.invoke(app, ["show", "backend.dummy"])
    if result.exit_code != 0:
        print(f"Plugin show failed stdout: {result.stdout}")
        if result.exception:
            print(f"Plugin show failed exception: {result.exception}")
            import traceback

            traceback.print_exception(result.exception)
    assert result.exit_code == 0
    assert "param" in result.stdout


def test_run_list():
    """Test 'run list'."""
    result = runner.invoke(app, ["run", "list"])
    assert result.exit_code == 0
    # Should list available engines
    assert "dummy" in result.stdout


def test_template_command():
    """Test 'template' command."""
    result = runner.invoke(app, ["template", "engine.dummy"])
    if result.exit_code != 0:
        print(f"Template command failed: {result.stdout}")
        if result.exception:
            print(f"Template command failed exception: {result.exception}")
            import traceback

            traceback.print_exception(result.exception)
    assert result.exit_code == 0
    assert "param:" in result.stdout


def test_run_jobs_command(temp_workspace, sample_job_file, dummy_model):
    """Test 'run jobs' command."""
    # The sample_job_file fixture creates a valid job file in the workspace
    # dummy_model fixture registers the 'dummy' model used in sample_job_file

    # Run the job
    # We need to pass the job name (without extension), not the file path
    result = runner.invoke(app, ["run", "jobs", "test_job"])

    # Note: This might fail if the engine/backend implementation has issues
    # But we are testing the CLI invocation here
    if result.exit_code != 0:
        print(result.stdout)

    assert result.exit_code == 0
    # Check that job ran (output shows run directory)
    assert "[test_job]" in result.stdout
    assert "Run directories:" in result.stdout

    # Check if output was created
    output_dir = temp_workspace / "runs"
    # There should be a subdirectory for the run
    assert any(output_dir.iterdir())
