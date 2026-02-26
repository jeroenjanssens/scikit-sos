"""Tests for CLI interface."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest


class TestCLI:
    """Test command-line interface."""

    @pytest.fixture
    def sample_data_file(self, tmp_path: Path) -> Path:
        """Create temporary CSV file with test data."""
        data = np.array([[1, 2], [3, 4], [5, 6], [100, 100]], dtype=float)
        filepath = tmp_path / 'data.csv'
        np.savetxt(filepath, data, delimiter=',')
        return filepath

    def test_cli_basic_usage(self, sample_data_file: Path) -> None:
        """Test basic CLI execution."""
        result = subprocess.run(
            ['sos', '-i', str(sample_data_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 4  # Four outlier scores

    def test_cli_with_perplexity(self, sample_data_file: Path) -> None:
        """Test CLI with custom perplexity."""
        result = subprocess.run(
            ['sos', '-p', '5', '-i', str(sample_data_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0

    def test_cli_with_threshold(self, sample_data_file: Path) -> None:
        """Test CLI with threshold for binary output."""
        result = subprocess.run(
            ['sos', '-t', '0.5', '-i', str(sample_data_file)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        # Should output 0s and 1s
        for line in lines:
            assert line in ['0', '1']

    def test_cli_stdin_stdout(self) -> None:
        """Test CLI with stdin/stdout pipes."""
        data = '1,2\n3,4\n5,6\n100,100\n'
        result = subprocess.run(
            ['sos'],
            input=data,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        lines = result.stdout.strip().split('\n')
        assert len(lines) == 4
