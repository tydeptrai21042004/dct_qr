"""Unified proposal/baseline benchmark layer.

This package deliberately sits above all embedding and extraction code. It does
not alter the mathematical rules of any proposal or baseline.
"""

from .adapters import BenchmarkAdapter, load_baseline_parameters
from .aggregation import aggregate_rows, flatten_rows, load_trials, write_aggregate
from .evaluator import evaluate_trial, key_size_bytes
from .registry import all_method_specs, get_method_spec, resolve_methods
from .types import BaselineRunParameters, BenchmarkMethodSpec, BenchmarkProtocol

__all__ = [
    "BenchmarkAdapter", "load_baseline_parameters", "aggregate_rows", "flatten_rows",
    "load_trials", "write_aggregate", "evaluate_trial", "key_size_bytes",
    "all_method_specs", "get_method_spec", "resolve_methods", "BaselineRunParameters",
    "BenchmarkMethodSpec", "BenchmarkProtocol",
]
