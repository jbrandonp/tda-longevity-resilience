"""Tests for config.py and logging_config.py — trivial sanity checks."""

import pytest


class TestConfig:
    def test_random_seed_is_int(self):
        from src.config import RANDOM_SEED
        assert isinstance(RANDOM_SEED, int)
        assert RANDOM_SEED == 42

    def test_pi_params_are_valid(self):
        from src.config import PI_SPREAD, PI_PIXELS
        assert PI_SPREAD > 0
        assert len(PI_PIXELS) == 2
        assert PI_PIXELS[0] > 0
        assert PI_PIXELS[1] > 0

    def test_ripser_config(self):
        from src.config import RIPSER_MAX_DIM, RIPSER_N_THREADS
        assert RIPSER_MAX_DIM >= 1
        assert RIPSER_N_THREADS >= -1  # -1 = auto-detect


class TestLoggingConfig:
    def test_get_logger_returns_logger(self):
        from src.logging_config import get_logger
        logger = get_logger("test_config")
        assert logger is not None
        logger.info("test message")

    def test_logger_namespace(self):
        from src.logging_config import get_logger
        logger = get_logger("sub.module")
        assert "sub.module" in logger.name
