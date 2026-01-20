"""Tests for CLI module."""

from __future__ import annotations

import pathlib
import tempfile

import pytest

from bpdecoderplus.cli import create_parser, main


class TestCreateParser:
    """Tests for create_parser function."""

    def test_parser_defaults(self):
        """Test parser has correct defaults."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.output == pathlib.Path("datasets/noisy_circuits")
        assert args.distance == 3
        assert args.rounds == [3, 5, 7]
        assert args.p == 0.01
        assert args.task == "z"
        assert args.no_smoke_test is False

    def test_custom_arguments(self):
        """Test parser with custom arguments."""
        parser = create_parser()
        args = parser.parse_args([
            "-d", "5",
            "-r", "1", "3",
            "-p", "0.02",
            "--task", "x",
            "--no-smoke-test",
        ])

        assert args.distance == 5
        assert args.rounds == [1, 3]
        assert args.p == 0.02
        assert args.task == "x"
        assert args.no_smoke_test is True


class TestMain:
    """Tests for main function."""

    def test_basic_generation(self):
        """Test basic circuit generation via CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "-o", tmpdir,
                "-d", "3",
                "-r", "3",
                "-p", "0.01",
                "--no-smoke-test",
            ])
            assert result == 0

            # Check output file exists
            output_file = pathlib.Path(tmpdir) / "sc_d3_r3_p0010_z.stim"
            assert output_file.exists()

    def test_multiple_rounds(self):
        """Test generation of multiple rounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "-o", tmpdir,
                "-r", "3", "5", "7",
                "--no-smoke-test",
            ])
            assert result == 0

            # Check all output files exist
            for r in [3, 5, 7]:
                output_file = pathlib.Path(tmpdir) / f"sc_d3_r{r}_p0010_z.stim"
                assert output_file.exists()

    def test_invalid_p_returns_error(self):
        """Test that invalid p returns error code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(["-o", tmpdir, "-p", "1.5"])
            assert result == 1

    def test_invalid_rounds_returns_error(self):
        """Test that invalid rounds returns error code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main(["-o", tmpdir, "-r", "0", "-1"])
            assert result == 1

    def test_x_task(self):
        """Test X-memory task generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "-o", tmpdir,
                "-r", "3",
                "--task", "x",
                "--no-smoke-test",
            ])
            assert result == 0

            output_file = pathlib.Path(tmpdir) / "sc_d3_r3_p0010_x.stim"
            assert output_file.exists()

    def test_with_smoke_test(self):
        """Test generation with smoke test enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = main([
                "-o", tmpdir,
                "-d", "3",
                "-r", "3",
                "-p", "0.01",
            ])
            assert result == 0
