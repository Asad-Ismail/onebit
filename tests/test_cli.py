"""Smoke tests for the CLI and module wiring.

These import every module and invoke each command's help so that broken imports
fail in CI instead of at the moment a user first runs the command.
"""

import importlib

import pytest
from click.testing import CliRunner

from onebit.cli import cli

MODULES = [
    "onebit",
    "onebit.cli",
    "onebit.engine",
    "onebit.generate",
    "onebit.bench",
    "onebit.models",
    "onebit.models.registry",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("command", ["run", "bench", "list", "info"])
def test_command_help(command):
    """Each command must at least render --help (imports its lazy deps)."""
    result = CliRunner().invoke(cli, [command, "--help"])
    assert result.exit_code == 0, result.output


def test_list_runs():
    result = CliRunner().invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "qwen2.5-3b" in result.output
