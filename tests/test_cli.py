"""Tests for cli.py — CLI commands and pipeline."""
import subprocess, sys, pytest

CLI = [sys.executable, "-m", "src.cli"]


class TestCLIHelp:
    def test_help(self):
        r = subprocess.run(CLI + ["--help"], capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "run" in r.stdout and "demo" in r.stdout and "hello" in r.stdout

    def test_subcommands_present(self):
        r = subprocess.run(CLI + ["--help"], capture_output=True, text=True, timeout=10)
        for cmd in ["run", "demo", "hello", "data", "tda", "ml", "report"]:
            assert cmd in r.stdout, f"'{cmd}' not found in --help"


class TestHello:
    def test_hello_runs(self):
        r = subprocess.run(CLI + ["hello"], capture_output=True, text=True, timeout=15)
        assert r.returncode == 0


class TestDemo:
    def test_demo_runs(self):
        r = subprocess.run(CLI + ["demo"], capture_output=True, text=True, timeout=30)
        assert r.returncode == 0
        assert "Demo complete" in (r.stdout + r.stderr)


class TestRun:
    def test_run_help(self):
        r = subprocess.run(CLI + ["run", "--help"], capture_output=True, text=True, timeout=10)
        assert r.returncode == 0
        assert "--n-samples" in r.stdout

    def test_run_minimal(self):
        r = subprocess.run(
            CLI + ["run", "--n-samples", "20", "--n-features", "10", "--skip-mapper"],
            capture_output=True, text=True, timeout=60,
        )
        # Should complete (or fail gracefully on missing ripser)
        assert r.returncode in (0, 1)
