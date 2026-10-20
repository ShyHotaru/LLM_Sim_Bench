"""LLM Sim Bench: simulate local LLM hardware and speed before you build."""

from .estimator import Efficiency, Estimate, estimate
from .mock import MockEngine
from .profiles import CPU, GPU, RAM, Model, System

__all__ = ["CPU", "GPU", "RAM", "Model", "System", "Efficiency", "Estimate", "estimate",
           "MockEngine"]
__version__ = "0.1.0"
