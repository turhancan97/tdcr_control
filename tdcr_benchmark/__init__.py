"""Python benchmark tools for tendon-driven continuum robot models."""

from tdcr_benchmark.config import RobotConfig, load_config
from tdcr_benchmark.runner import run_benchmark

__all__ = ["RobotConfig", "load_config", "run_benchmark"]

