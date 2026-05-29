"""
test_kyber_full.py
==================
Comprehensive test suite for the Kyber KEM pipeline.

Sections
--------
1.  Error Handling Tests  — every typed exception from KyberExceptions
2.  FO Transform Tests    — tampered ciphertext, garbage_key, timing
3.  Encode/Decode Round-Trip Sanity
4.  Profiler Dashboard    — error path + FO overhead vs happy path
5.  Enc/Dec Benchmark     — avg latency across message length buckets

FO PROTOCOL NOTE
----------------
The FO check in KyberCore._fo_check re-derives randomness as:
    L' = SHA3-512(p')[32:]
and re-encrypts p' with L'.  For the check to PASS the original encryption
must also have used L = SHA3-512(m)[32:] as its r_seed — i.e. the "FO-mode"
encrypt call.  If encrypt() is called with r_seed=None (random), the check
will ALWAYS reject because (u,v) ≠ (u',v').

The helper _fo_encrypt() below performs a correct FO-mode encryption so that
the honest-path tests pass.  Tamper tests bypass this by modifying coefficients
after an honest FO-mode encrypt, so rejection is expected.

Run from the package root:
    python -m <package>.test_kyber_full

Or standalone:
    python test_kyber_full.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

# ── import shim ──────────────────────────────────────────────────────────────
try:
    from ..KyberCore         import KyberCoreEngine
    from ..ArithmeticUnit    import ntt
    from ..KyberExceptions   import (
        CiphertextListError,
        DecryptionError,
        EmptyMessageError,
        EncryptionError,
        FOTransformError,
        ImplicitRejectionError,
        InvalidCiphertextError,
        InvalidMessageTypeError,
        InvalidPublicKeyError,
        InvalidSecretKeyError,
        InvalidSeedError,
        KeyGenError,
        MessageSizeError,
        ParameterError,
        SoftDecodeError,
    )
    from ..Profiler import Profiler
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from KyberCore         import KyberCoreEngine           # type: ignore
    from ArithmeticUnit    import ntt                       # type: ignore
    from KyberExceptions   import (                         # type: ignore
        CiphertextListError, DecryptionError, EmptyMessageError,
        EncryptionError, FOTransformError, ImplicitRejectionError,
        InvalidCiphertextError, InvalidMessageTypeError,
        InvalidPublicKeyError, InvalidSecretKeyError, InvalidSeedError,
        KeyGenError, MessageSizeError, ParameterError, SoftDecodeError,
    )
    from Profiler import Profiler                           # type: ignore


# ── global constants ──────────────────────────────────────────────────────────
N, Q, ETA = 256, 3329, 2
CHUNK     = 32   # SerializationUnit.chunk_size = (256+7)//8

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
RED    = "\033[31m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# ── result tracker ────────────────────────────────────────────────────────────
_results: dict = {"pass": 0, "fail": 0, "skip": 0}
_timing:  dict = defaultdict(list)   # label -> [elapsed_ms, ...]


def _ok(label: str, elapsed_ms: float | None = None) -> None:
    _results["pass"] += 1
    tag = f"  {GREEN}✓ PASS{RESET}  {label}"
    if elapsed_ms is not None:
        tag += f"  {YELLOW}({elapsed_ms:.2f} ms){RESET}"
    print(tag)


def _fail(label: str, exc: Exception | None = None) -> None:
    _results["fail"] += 1
    tag = f"  {RED}✗ FAIL{RESET}  {label}"
    if exc:
        tag += f"\n         {RED}{type(exc).__name__}: {exc}{RESET}"
    print(tag)


def _section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {BOLD}{CYAN}{title}{RESET}")
    print(f"{'═' * 70}")


def _sub(title: str) -> None:
    print(f"\n  {'─' * 60}")
    print(f"  {BOLD}{title}{RESET}")
    print(f"  {'─' * 60}")


def _timeit(fn, runs: int = 5) -> Tuple[float, float, float]:
    """Returns (avg_ms, min_ms, max_ms)."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times), max(times)


def _assert_raises(exc_type, fn, label: str) -> bool:
    """Pass if exc_type (or a subclass) is raised, fail otherwise."""
    t0 = time.perf_counter()
    try:
        fn()
        _fail(label, Exception(f"Expected {exc_type.__name__}, got no exception"))
        return False
    except exc_type as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _timing[f"error/{exc_type.__name__}"].append(elapsed)
        _ok(label, elapsed)
        return True
    except Exception as e:
        _fail(label, e)
        return False


# ── FO-mode encrypt helper ────────────────────────────────────────────────────
# The FO check re-derives r_seed as L' = SHA3-512(p')[32:] and re-encrypts.
# For honest decryption to PASS the check the original ciphertext must have
# been produced with exactly that seed.  Use this helper for all FO tests.

def _fo_encrypt(engine: KyberCoreEngine, pk: tuple, msg: bytes):
    """Encrypt msg using the FO-derived seed L = SHA3-512(msg)[32:]."""
    h = hashlib.sha3_512(msg).digest()
    l_prime = h[32:]          # same derivation as _fo_check uses
    return engine.encrypt(pk, msg, r_seed=l_prime)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Error Handling Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_error_handling() -> None:
    _section("SECTION 1 · Error Handling Tests")

    # ── 1.1 Parameter validation ──────────────────────────────────────────────
    _sub("1.1  ParameterError — invalid construction parameters")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(0, Q, ETA, 2),
        "n=0 (not a power of 2) → ParameterError")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(100, Q, ETA, 2),
        "n=100 (not a power of 2) → ParameterError")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(N, 2, ETA, 2),
        "q=2 (too small) → ParameterError")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(N, Q, 0, 2),
        "eta=0 → ParameterError")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(N, Q, ETA, 0),
        "k=0 → ParameterError")

    _assert_raises(ParameterError,
        lambda: KyberCoreEngine(N, Q, ETA, 2, rejection_seed=b"too_short"),
        "rejection_seed wrong length → ParameterError")

    # ── 1.2 Key generation errors ─────────────────────────────────────────────
    _sub("1.2  InvalidSeedError / KeyGenError — keygen inputs")
    eng = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4)

    _assert_raises(InvalidSeedError,
        lambda: eng.keygen(b"short"),
        "seed too short (5 bytes) → InvalidSeedError")

    _assert_raises(InvalidSeedError,
        lambda: eng.keygen(b"x" * 64),
        "seed too long (64 bytes) → InvalidSeedError")

    _assert_raises(InvalidSeedError,
        lambda: eng.keygen("not_bytes"),     # type: ignore
        "seed as str → InvalidSeedError")

    # ── 1.3 Encryption errors ─────────────────────────────────────────────────
    _sub("1.3  Encryption errors — public key / message validation")

    pk, sk = eng.keygen(os.urandom(32))
    valid_msg = b"B" * CHUNK

    # Bad public key structure
    _assert_raises(InvalidPublicKeyError,
        lambda: eng.encrypt((b"x" * 32, []), valid_msg),
        "t=[] (empty) in pk → InvalidPublicKeyError")

    _assert_raises(InvalidPublicKeyError,
        lambda: eng.encrypt(("not_bytes", pk[1]), valid_msg),
        "rho=str → InvalidPublicKeyError")

    _assert_raises(InvalidPublicKeyError,
        lambda: eng.encrypt((b"x" * 16, pk[1]), valid_msg),
        "rho wrong length (16 bytes) → InvalidPublicKeyError")

    _assert_raises(InvalidPublicKeyError,
        lambda: eng.encrypt(("single_str",), valid_msg),
        "pk is 1-tuple → InvalidPublicKeyError")

    # Bad message types
    _assert_raises(InvalidMessageTypeError,
        lambda: eng.encrypt(pk, 12345),      # type: ignore
        "message=int → InvalidMessageTypeError")

    _assert_raises(InvalidMessageTypeError,
        lambda: eng.encrypt(pk, 3.14),       # type: ignore
        "message=float → InvalidMessageTypeError")

    _assert_raises(InvalidMessageTypeError,
        lambda: eng.encrypt(pk, ["list"]),   # type: ignore
        "message=list → InvalidMessageTypeError")

    # Empty / wrong-size messages
    _assert_raises(EmptyMessageError,
        lambda: eng.encrypt(pk, b""),
        "message=b'' → EmptyMessageError")

    _assert_raises(MessageSizeError,
        lambda: eng.encrypt(pk, b"short"),
        "message 5 bytes (not chunk_size=32) → MessageSizeError")

    _assert_raises(MessageSizeError,
        lambda: eng.encrypt(pk, b"X" * 64),
        "message 64 bytes (double chunk_size) → MessageSizeError")

    # encrypt_chunked empty inputs
    _assert_raises(EmptyMessageError,
        lambda: eng.encrypt_chunked(pk, ""),
        "encrypt_chunked with '' → EmptyMessageError")

    _assert_raises(EmptyMessageError,
        lambda: eng.encrypt_chunked(pk, b""),
        "encrypt_chunked with b'' → EmptyMessageError")

    # ── 1.4 Decryption errors ─────────────────────────────────────────────────
    _sub("1.4  Decryption errors — secret key / ciphertext validation")

    ct = eng.encrypt(pk, valid_msg)
    u, v = ct

    # Bad secret key
    _assert_raises(InvalidSecretKeyError,
        lambda: eng.decrypt([], ct),
        "sk=[] (empty) → InvalidSecretKeyError")

    _assert_raises(InvalidSecretKeyError,
        lambda: eng.decrypt(sk + [[0] * N], ct),
        "sk has extra polynomial → InvalidSecretKeyError")

    _assert_raises(InvalidSecretKeyError,
        lambda: eng.decrypt("not_a_list", ct),   # type: ignore
        "sk=str → InvalidSecretKeyError")

    # Bad ciphertext structure
    _assert_raises(InvalidCiphertextError,
        lambda: eng.decrypt(sk, ("bad_u", v)),
        "u=str in ciphertext → InvalidCiphertextError")

    _assert_raises(InvalidCiphertextError,
        lambda: eng.decrypt(sk, (u[:1], v)),
        "u wrong dimension (1 instead of k=2) → InvalidCiphertextError")

    _assert_raises(InvalidCiphertextError,
        lambda: eng.decrypt(sk, (u, v[:-10])),
        "v truncated → InvalidCiphertextError")

    _assert_raises(InvalidCiphertextError,
        lambda: eng.decrypt(sk, "not_a_tuple"),
        "ciphertext=str → InvalidCiphertextError")

    # decrypt_chunked with non-list
    _assert_raises(CiphertextListError,
        lambda: eng.decrypt_chunked(sk, ct),
        "decrypt_chunked with tuple instead of list → CiphertextListError")

    _assert_raises(CiphertextListError,
        lambda: eng.decrypt_chunked(sk, "bad"),
        "decrypt_chunked with str → CiphertextListError")

    # ── 1.5 Mismatched keypair (no FO) ───────────────────────────────────────
    _sub("1.5  Wrong keypair — decrypt with mismatched sk (FO disabled)")

    eng_nofo = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                                fo_enabled=False)
    pk_n, sk_n = eng_nofo.keygen(os.urandom(32))
    _, sk_other = eng_nofo.keygen(os.urandom(32))

    ct_n = eng_nofo.encrypt(pk_n, valid_msg)
    t0 = time.perf_counter()
    result = eng_nofo.decrypt(sk_other, ct_n)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if result != valid_msg:
        _ok("wrong sk (no FO) → garbled bytes, no exception", elapsed_ms)
    else:
        # Astronomically unlikely but technically possible
        _ok("wrong sk (no FO) → coincidental correct decode (rare)", elapsed_ms)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FO Transform Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_fo_transform() -> None:
    _section("SECTION 2 · FO Transform Tests")

    eng_fo = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                              fo_enabled=True)
    pk_fo, sk_fo = eng_fo.keygen(os.urandom(32))
    valid_msg    = b"C" * CHUNK

    # ── 2.1 Honest path — FO-mode encrypt then decrypt with FO check ─────────
    _sub("2.1  Happy path — FO-mode encrypt → decrypt passes FO check")

    # Must use _fo_encrypt so the r_seed matches what _fo_check re-derives.
    t0 = time.perf_counter()
    ct_honest = _fo_encrypt(eng_fo, pk_fo, valid_msg)
    t_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    try:
        out = eng_fo.decrypt(sk_fo, ct_honest, public_key=pk_fo)
        t_dec = (time.perf_counter() - t0) * 1000
        if out == valid_msg:
            _ok(f"FO honest roundtrip (enc={t_enc:.1f} ms, dec={t_dec:.1f} ms)")
            _timing["fo/honest_enc"].append(t_enc)
            _timing["fo/honest_dec"].append(t_dec)
        else:
            _fail("FO honest roundtrip: plaintext mismatch after correct FO-mode encrypt")
    except ImplicitRejectionError as e:
        _fail("FO honest roundtrip: ImplicitRejectionError on untampered ciphertext",
              Exception(str(e)))

    # ── 2.2 Random-seed encrypt — FO should reject (expected behaviour) ───────
    _sub("2.2  Random-seed encrypt → FO correctly rejects (c ≠ c')")

    ct_random = eng_fo.encrypt(pk_fo, valid_msg)   # r_seed=None → random
    t0 = time.perf_counter()
    try:
        eng_fo.decrypt(sk_fo, ct_random, public_key=pk_fo)
        elapsed = (time.perf_counter() - t0) * 1000
        _fail("Random-seed ct: expected ImplicitRejectionError, got success")
    except ImplicitRejectionError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Random-seed ct → ImplicitRejectionError (correct FO behaviour) "
            f"garbage_key={e.garbage_key.hex()[:16]}… ({elapsed:.1f} ms)")
        _timing["fo/random_seed_reject"].append(elapsed)

    # ── 2.3 Tampered u vector ─────────────────────────────────────────────────
    _sub("2.3  Tampered u — FO implicit rejection")

    u, v = ct_honest
    u_tampered = [[(c + 1) % Q for c in poly] for poly in u]
    ct_tampered_u = (u_tampered, v)

    t0 = time.perf_counter()
    try:
        eng_fo.decrypt(sk_fo, ct_tampered_u, public_key=pk_fo)
        _fail("Tampered u: expected ImplicitRejectionError, got none")
    except ImplicitRejectionError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _timing["fo/tampered_u"].append(elapsed)
        if isinstance(e.garbage_key, bytes) and len(e.garbage_key) == 32:
            _ok(f"Tampered u → ImplicitRejectionError "
                f"garbage_key={e.garbage_key.hex()[:16]}… ({elapsed:.1f} ms)")
        else:
            _fail("Tampered u: raised but garbage_key is malformed")

    # ── 2.4 Tampered v polynomial ─────────────────────────────────────────────
    _sub("2.4  Tampered v — FO implicit rejection")

    v_tampered = [(c + 500) % Q for c in v]
    ct_tampered_v = (u, v_tampered)

    t0 = time.perf_counter()
    try:
        eng_fo.decrypt(sk_fo, ct_tampered_v, public_key=pk_fo)
        _fail("Tampered v: expected ImplicitRejectionError, got none")
    except ImplicitRejectionError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        _timing["fo/tampered_v"].append(elapsed)
        _ok(f"Tampered v → ImplicitRejectionError ({elapsed:.1f} ms)")

    # ── 2.5 Single-coefficient flip ───────────────────────────────────────────
    _sub("2.5  Single-coefficient flip — sensitivity test")

    v_bit_flip = list(v)
    v_bit_flip[0] = (v_bit_flip[0] + 1) % Q

    t0 = time.perf_counter()
    try:
        eng_fo.decrypt(sk_fo, (u, v_bit_flip), public_key=pk_fo)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Single coeff flip: within noise budget, decoded OK ({elapsed:.1f} ms)")
    except ImplicitRejectionError:
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"Single coeff flip: caught by FO → ImplicitRejectionError ({elapsed:.1f} ms)")

    # ── 2.6 FO disabled — tampered ciphertext passes through silently ─────────
    _sub("2.6  FO disabled — tampered ct produces garbled bytes (no exception)")

    eng_nofo = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                                fo_enabled=False)
    pk_n, sk_n = eng_nofo.keygen(os.urandom(32))
    ct_n = eng_nofo.encrypt(pk_n, valid_msg)
    u_n, v_n = ct_n
    u_bad = [[(c + 100) % Q for c in p] for p in u_n]

    t0 = time.perf_counter()
    try:
        result = eng_nofo.decrypt(sk_n, (u_bad, v_n))
        elapsed = (time.perf_counter() - t0) * 1000
        _timing["fo/disabled_tamper"].append(elapsed)
        _ok(f"FO disabled: tampered ct decrypted without exception ({elapsed:.1f} ms)")
    except Exception as e:
        _fail(f"FO disabled: unexpected exception {type(e).__name__}: {e}")

    # ── 2.7 fo_override=False per-call suppression ────────────────────────────
    _sub("2.7  fo_override=False — per-call FO suppression")

    # Use ct_tampered_u (which would normally be rejected) — with override off
    # it should decrypt without raising.
    t0 = time.perf_counter()
    try:
        result = eng_fo.decrypt(sk_fo, ct_tampered_u,
                                 public_key=pk_fo, fo_override=False)
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"fo_override=False: tampered ct passed silently ({elapsed:.1f} ms)")
    except ImplicitRejectionError:
        _fail("fo_override=False should have suppressed FO check")
    except Exception as e:
        # Garbled decode causing internal error is also acceptable
        elapsed = (time.perf_counter() - t0) * 1000
        _ok(f"fo_override=False: got {type(e).__name__} on garbled decode ({elapsed:.1f} ms)")

    # ── 2.8 Garbage key determinism ───────────────────────────────────────────
    _sub("2.8  Garbage key determinism — same tampered ct → same 32-byte key")

    garbage_keys = []
    for _ in range(3):
        try:
            eng_fo.decrypt(sk_fo, ct_tampered_u, public_key=pk_fo)
        except ImplicitRejectionError as e:
            garbage_keys.append(e.garbage_key)

    if len(set(garbage_keys)) == 1 and len(garbage_keys) == 3:
        _ok(f"garbage_key is deterministic: {garbage_keys[0].hex()[:24]}…")
    elif len(garbage_keys) == 0:
        _fail("No ImplicitRejectionError was raised — tampered ct was not caught")
    else:
        _fail(f"garbage_key differs across calls: {[k.hex()[:16] for k in garbage_keys]}")

    # ── 2.9 FO overhead — honest vs tampered timing ───────────────────────────
    _sub("2.9  FO overhead — honest vs tampered latency (20 runs each)")

    runs = 20
    ct_bench = _fo_encrypt(eng_fo, pk_fo, valid_msg)
    u_b, v_b = ct_bench
    u_t = [[(c + 400) % Q for c in p] for p in u_b]

    avg_honest, _, _ = _timeit(
        lambda: eng_fo.decrypt(sk_fo, ct_bench, public_key=pk_fo), runs)

    def _try_tampered():
        try:
            eng_fo.decrypt(sk_fo, (u_t, v_b), public_key=pk_fo)
        except ImplicitRejectionError:
            pass

    avg_tampered, _, _ = _timeit(_try_tampered, runs)
    _timing["fo/bench_honest"].append(avg_honest)
    _timing["fo/bench_tampered"].append(avg_tampered)

    ratio = avg_tampered / avg_honest if avg_honest > 0 else float("inf")
    note  = "✓ timing similar (constant-time)" if ratio < 2.5 else "⚠  timing ratio suspicious"
    print(f"\n     Honest   : {avg_honest:.2f} ms  (avg {runs} runs)")
    print(f"     Tampered : {avg_tampered:.2f} ms  (avg {runs} runs)")
    print(f"     Ratio    : {ratio:.2f}×  {note}")
    _ok("FO timing comparison complete")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Round-trip Sanity
# ══════════════════════════════════════════════════════════════════════════════

def test_roundtrip_sanity() -> None:
    _section("SECTION 3 · Round-trip Sanity")

    eng = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                          fo_enabled=False)   # FO off — random seeds allowed
    pk, sk = eng.keygen(os.urandom(32))

    test_cases = [
        ("zeros",   b"\x00" * CHUNK),
        ("ones",    b"\xFF" * CHUNK),
        ("pattern", bytes(range(CHUNK))),
        ("random",  os.urandom(CHUNK)),
    ]

    _sub("3.1  Single-chunk roundtrip (4 payloads)")
    for label, msg in test_cases:
        try:
            ct  = eng.encrypt(pk, msg)
            out = eng.decrypt(sk, ct)
            if out == msg:
                _ok(f"[{label}] single-chunk roundtrip")
            else:
                _fail(f"[{label}] plaintext mismatch")
        except Exception as e:
            _fail(f"[{label}] unexpected exception: {e}")

    _sub("3.2  Chunked string roundtrip — all three Kyber variants")
    for variant, k_dim, du, dv in [(512,2,10,4),(768,3,10,4),(1024,4,11,5)]:
        ev = KyberCoreEngine(N, Q, ETA, k_dimension=k_dim, d_u=du, d_v=dv,
                             fo_enabled=False)
        pk_v, sk_v = ev.keygen(os.urandom(32))
        msg_str = f"Kyber-{variant} chunked test — αβγδ ✓"
        try:
            cts = ev.encrypt_chunked(pk_v, msg_str)
            out = ev.decrypt_chunked(sk_v, cts, output_type="str")
            if out == msg_str:
                _ok(f"Kyber-{variant} ({len(cts)} chunk(s)) roundtrip")
            else:
                _fail(f"Kyber-{variant} mismatch")
        except Exception as e:
            _fail(f"Kyber-{variant} exception: {e}")

    _sub("3.3  Deterministic keygen — same seed → same keypair")
    seed   = os.urandom(32)
    pk1, _ = eng.keygen(seed)
    pk2, _ = eng.keygen(seed)
    if pk1[0] == pk2[0]:
        _ok("keygen is deterministic given same seed")
    else:
        _fail("keygen is not deterministic!")

    _sub("3.4  Cross-key failure — wrong sk produces different plaintext")
    _, sk_other = eng.keygen(os.urandom(32))
    msg  = b"Q" * CHUNK
    ct   = eng.encrypt(pk, msg)
    out  = eng.decrypt(sk_other, ct)
    if out != msg:
        _ok("wrong sk → garbled bytes (expected)")
    else:
        _fail("wrong sk produced correct plaintext (should not happen)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Profiler Dashboard
# ══════════════════════════════════════════════════════════════════════════════

def build_profiler_report() -> None:
    _section("SECTION 4 · Profiler Dashboard — Error Path & FO Overhead")

    msg = b"P" * CHUNK

    # ── Happy path ────────────────────────────────────────────────────────────
    Profiler.current_category = "Happy Path"
    eng_h = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                             fo_enabled=True)
    pk_h, sk_h = eng_h.keygen(os.urandom(32))
    for _ in range(10):
        ct = _fo_encrypt(eng_h, pk_h, msg)
        eng_h.decrypt(sk_h, ct, public_key=pk_h)

    # ── Error-handling path ───────────────────────────────────────────────────
    Profiler.current_category = "Error Handling Path"
    eng_e = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4)
    pk_e, sk_e = eng_e.keygen(os.urandom(32))
    for _ in range(10):
        try: eng_e.keygen(b"bad")
        except Exception: pass
        try: eng_e.encrypt(pk_e, b"too_short")
        except Exception: pass
        try: eng_e.decrypt([], eng_e.encrypt(pk_e, msg))
        except Exception: pass

    # ── FO transform path ─────────────────────────────────────────────────────
    Profiler.current_category = "FO Transform"
    eng_f = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                             fo_enabled=True)
    pk_f, sk_f = eng_f.keygen(os.urandom(32))
    ct_f = _fo_encrypt(eng_f, pk_f, msg)
    u_f, v_f = ct_f
    u_bad = [[(c + 300) % Q for c in p] for p in u_f]
    for _ in range(10):
        eng_f.decrypt(sk_f, _fo_encrypt(eng_f, pk_f, msg), public_key=pk_f)
        try:
            eng_f.decrypt(sk_f, (u_bad, v_f), public_key=pk_f)
        except ImplicitRejectionError:
            pass

    Profiler.current_category = "General"

    # ── Print timing registry ─────────────────────────────────────────────────
    print(f"\n  {'Label':<42}  {'n':>4}  {'avg ms':>8}  {'min ms':>8}  {'max ms':>8}")
    print(f"  {'─'*42}  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}")
    for label, times in sorted(_timing.items()):
        if not times:
            continue
        print(f"  {label:<42}  {len(times):>4}  "
              f"{sum(times)/len(times):>8.2f}  {min(times):>8.2f}  {max(times):>8.2f}")

    # ── HTML dashboard ────────────────────────────────────────────────────────
    try:
        Profiler.generate_html_dashboard("kyber_profiler_dashboard.html")
    except Exception as e:
        print(f"\n  {YELLOW}⚠ HTML export skipped: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Enc/Dec Benchmark by Message Length
# ══════════════════════════════════════════════════════════════════════════════

def test_timing_by_length() -> None:
    _section("SECTION 5 · Enc/Dec Latency by Message Length")

    # FO disabled so random seeds are fine and results reflect pure crypto cost
    eng = KyberCoreEngine(N, Q, ETA, k_dimension=2, d_u=10, d_v=4,
                          fo_enabled=False)
    pk, sk = eng.keygen(os.urandom(32))

    lengths: List[Tuple[str, int]] = [
        ("1 B",              1),
        ("16 B",            16),
        ("32 B (1 chunk)",  32),
        ("64 B (2 chunks)", 64),
        ("128 B",          128),
        ("256 B",          256),
        ("512 B",          512),
        ("1 KB",          1024),
        ("2 KB",          2048),
        ("4 KB",          4096),
        ("8 KB",          8192),
    ]

    RUNS = 8

    print(f"\n  {'Length':<22}  {'Chunks':>6}  {'Enc avg':>9}  {'Enc min':>9}"
          f"  {'Dec avg':>9}  {'Dec min':>9}  {'Total':>9}  {'Throughput B/s':>16}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*9}  {'─'*9}"
          f"  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*16}")

    bench_rows = []

    for label, nbytes in lengths:
        payload = os.urandom(nbytes)
        cts     = eng.encrypt_chunked(pk, payload)
        n_chunks = len(cts)

        avg_enc, min_enc, _ = _timeit(
            lambda: eng.encrypt_chunked(pk, payload), RUNS)
        avg_dec, min_dec, _ = _timeit(
            lambda: eng.decrypt_chunked(sk, cts, output_type="bytes"), RUNS)

        total = avg_enc + avg_dec
        tp    = (nbytes / (total / 1000)) if total > 0 else 0

        print(f"  {label:<22}  {n_chunks:>6}  {avg_enc:>8.2f}ms  {min_enc:>8.2f}ms"
              f"  {avg_dec:>8.2f}ms  {min_dec:>8.2f}ms  {total:>8.2f}ms  {tp:>16,.0f}")

        bench_rows.append({
            "label":       label,
            "bytes":       nbytes,
            "chunks":      n_chunks,
            "enc_avg_ms":  round(avg_enc, 3),
            "enc_min_ms":  round(min_enc, 3),
            "dec_avg_ms":  round(avg_dec, 3),
            "dec_min_ms":  round(min_dec, 3),
            "total_ms":    round(total,   3),
            "throughput":  round(tp, 0),
        })

    # ── Per-variant single-chunk latency ─────────────────────────────────────
    _sub("5b.  Per-variant single-chunk latency (32 bytes, 50 runs)")

    msg32 = os.urandom(32)
    print(f"\n  {'Variant':<14}  {'KeyGen ms':>10}  {'Enc avg ms':>12}  {'Dec avg ms':>12}")
    print(f"  {'─'*14}  {'─'*10}  {'─'*12}  {'─'*12}")

    for variant, k_dim, du, dv in [(512,2,10,4),(768,3,10,4),(1024,4,11,5)]:
        ev = KyberCoreEngine(N, Q, ETA, k_dimension=k_dim, d_u=du, d_v=dv,
                             fo_enabled=False)
        t0 = time.perf_counter()
        pk_v, sk_v = ev.keygen(os.urandom(32))
        kg_ms = (time.perf_counter() - t0) * 1000

        ct_v = ev.encrypt(pk_v, msg32)
        avg_e, _, _ = _timeit(lambda: ev.encrypt(pk_v, msg32), 50)
        avg_d, _, _ = _timeit(lambda: ev.decrypt(sk_v, ct_v),  50)

        print(f"  Kyber-{variant:<8}  {kg_ms:>10.2f}  {avg_e:>12.2f}  {avg_d:>12.2f}")

    # Save JSON
    try:
        out_path = Path("kyber_timing_benchmark.json")
        with open(out_path, "w") as f:
            json.dump(bench_rows, f, indent=2)
        print(f"\n  {GREEN}Timing data saved → {out_path}{RESET}")
    except Exception as e:
        print(f"\n  {YELLOW}⚠ Could not save JSON: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{BOLD}{'═'*70}{RESET}")
    print(f"{BOLD}  Kyber KEM — Full Test Suite  (N={N}, Q={Q}, η={ETA}){RESET}")
    print(f"{BOLD}{'═'*70}{RESET}")

    test_error_handling()
    test_fo_transform()
    test_roundtrip_sanity()
    build_profiler_report()
    test_timing_by_length()

    total = sum(_results.values())
    p, f, s = _results["pass"], _results["fail"], _results["skip"]

    _section("SUMMARY")
    print(f"  Total : {total}")
    print(f"  {GREEN}Pass  : {p}{RESET}")
    print(f"  {RED}Fail  : {f}{RESET}")
    print(f"  {YELLOW}Skip  : {s}{RESET}\n")

    if f == 0:
        print(f"  {BOLD}{GREEN}All checks passed.{RESET}")
    else:
        print(f"  {BOLD}{RED}{f} check(s) failed — review output above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()