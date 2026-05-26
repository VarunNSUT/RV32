"""
helpers.py — shared fixtures, timing utilities, and pretty-printer for the
             Kyber KEM test suite.
"""

import os
import struct
import time
from functools import wraps
from typing import Callable

from ..KyberCore import KyberCoreEngine

# ── standard parameter sets ──────────────────────────────────────────────────

N   = 256
Q   = 3329
ETA = 2

VARIANTS = {
    "kyber512":  {"k": 2},
    "kyber768":  {"k": 3},
    "kyber1024": {"k": 4},
}


def make_engine(k: int = 2, compress: bool = False, **kwargs) -> KyberCoreEngine:
    return KyberCoreEngine(N, Q, ETA, k_dimension=k, compress=compress, **kwargs)


def make_keypair(k: int = 2):
    eng = make_engine(k)
    return eng, *eng.keygen()


# ── timing ───────────────────────────────────────────────────────────────────

_bench_log: list = []     # accumulated by BenchmarkMixin; printed at the end


def timed(label: str):
    """Decorator that records wall-time of a test method into _bench_log."""
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0     = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            _bench_log.append((label, elapsed))
            return result
        return wrapper
    return decorator


# ── display helpers ──────────────────────────────────────────────────────────

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m⊘ SKIP\033[0m"


def section(title: str) -> None:
    bar = "─" * (60 - len(title) - 2)
    print(f"\n── {title} {bar}")


def print_bench_report() -> None:
    if not _bench_log:
        return
    print("\n" + "=" * 65)
    print("  BENCHMARK REPORT")
    print("=" * 65)
    print(f"  {'Test':<48} {'ms':>8}")
    print(f"  {'-'*48} {'-'*8}")
    for label, ms in _bench_log:
        print(f"  {label:<48} {ms:>8.2f}")
    total = sum(ms for _, ms in _bench_log)
    print(f"  {'─'*48} {'─'*8}")
    print(f"  {'TOTAL':<48} {total:>8.2f}")
    print()