"""
Performance benchmarks for Agent Zero speed optimizations.

This module measures baseline performance and validates optimization gains.
"""

from .benchmarks import (
    BenchmarkSuite,
    run_all_benchmarks,
    generate_report
)

__all__ = [
    'BenchmarkSuite',
    'run_all_benchmarks',
    'generate_report'
]
