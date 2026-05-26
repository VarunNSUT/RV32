"""
test_kyber.py
─────────────
Comprehensive test suite for the Kyber KEM pipeline.

Run with:
    python -m Kyber_KEM.tests          (from the repo root)
    python -m pytest Kyber_KEM/tests   (if pytest is installed)

Sections
────────
  1.  Core correctness        – encrypt/decrypt round-trips
  2.  String payloads         – short, unicode, long, empty-ish
  3.  Integer payloads        – small, large, negative proxy, zero
  4.  Float payloads          – normal, edge (inf, nan)
  5.  Bytes payloads          – random, all-zeros, all-ones, structured
  6.  Chunked API             – multi-chunk encrypt/decrypt round-trips
  7.  Compression             – ratio checks, round-trips, tamper detection
  8.  Error handling          – every exception type in KyberExceptions
  9.  Multi-variant           – Kyber-512 / 768 / 1024
  10. Noise analysis          – coefficient-level noise statistics
  11. Benchmarks              – per-operation timing
"""

import os
import struct
import time
import unittest
from typing import List

# ── imports from the package ─────────────────────────────────────────────────
from ..ArithmeticUnit import Poly
from ..CiphertextCompressor import CiphertextCompressor
from ..KyberCore import KyberCoreEngine, _to_bytes
from ..KyberExceptions import (
    CiphertextListError,
    DecryptionError,
    DecompressionError,
    EmptyMessageError,
    EncryptionError,
    InvalidCiphertextError,
    InvalidMessageTypeError,
    InvalidPublicKeyError,
    InvalidSecretKeyError,
    InvalidSeedError,
    KeyGenError,
    MessageSizeError,
    ParameterError,
)
from ..SerializationUnit import SerializationUnit
from ..Profiler import Profiler
from .helpers import (
    N, Q, ETA,
    VARIANTS, PASS, FAIL,
    make_engine, make_keypair,
    section, _bench_log,
)

CHUNK = (N + 7) // 8   # 32 bytes


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across test cases
# ─────────────────────────────────────────────────────────────────────────────

def _roundtrip_str(eng, pk, sk, text: str) -> bool:
    cts = eng.encrypt_chunked(pk, text)
    # output_type="str" used as fallback; tagged payloads auto-detect
    return eng.decrypt_chunked(sk, cts, output_type="str") == text


def _roundtrip_bytes(eng, pk, sk, data: bytes) -> bool:
    cts = eng.encrypt_chunked(pk, data)
    return eng.decrypt_chunked(sk, cts, output_type="bytes") == data


def _roundtrip_int(eng, pk, sk, value: int) -> bool:
    cts = eng.encrypt_chunked(pk, value)
    return eng.decrypt_chunked(sk, cts, output_type="int") == value


def _roundtrip_float(eng, pk, sk, value: float) -> bool:
    import math
    cts = eng.encrypt_chunked(pk, value)
    rec = eng.decrypt_chunked(sk, cts, output_type="float")
    if math.isnan(value):
        return math.isnan(rec)
    if math.isinf(value):
        return math.isinf(rec) and (value > 0) == (rec > 0)
    return rec == value


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestCoreCorrectness(unittest.TestCase):

    def setUp(self):
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def test_01_keygen_returns_tuple(self):
        self.assertIsInstance(self.pk, tuple)
        self.assertEqual(len(self.pk), 2)

    def test_02_keygen_pk_structure(self):
        rho, t = self.pk
        self.assertEqual(len(rho), 32)
        self.assertIsInstance(t, list)
        self.assertEqual(len(t), 2)   # k=2

    def test_03_keygen_sk_structure(self):
        self.assertIsInstance(self.sk, list)
        self.assertEqual(len(self.sk), 2)

    def test_04_single_chunk_roundtrip(self):
        msg = os.urandom(CHUNK)
        ct  = self.eng.encrypt(self.pk, msg)
        rec = self.eng.decrypt(self.sk, ct)
        self.assertEqual(rec, msg)

    def test_05_all_zeros_chunk(self):
        msg = b"\x00" * CHUNK
        ct  = self.eng.encrypt(self.pk, msg)
        self.assertEqual(self.eng.decrypt(self.sk, ct), msg)

    def test_06_all_ones_chunk(self):
        msg = b"\xff" * CHUNK
        ct  = self.eng.encrypt(self.pk, msg)
        self.assertEqual(self.eng.decrypt(self.sk, ct), msg)

    def test_07_deterministic_keygen(self):
        seed = os.urandom(32)
        pk1, sk1 = self.eng.keygen(seed)
        pk2, sk2 = self.eng.keygen(seed)
        self.assertEqual(pk1[0], pk2[0])   # rho identical
        self.assertEqual(sk1, sk2)

    def test_08_different_seeds_different_keys(self):
        pk1, _ = self.eng.keygen(os.urandom(32))
        pk2, _ = self.eng.keygen(os.urandom(32))
        self.assertNotEqual(pk1[0], pk2[0])

    def test_09_ciphertext_not_plaintext(self):
        msg = b"A" * CHUNK
        ct  = self.eng.encrypt(self.pk, msg)
        u, v = ct
        # u and v are polynomial vectors — not equal to the plaintext directly
        flat = [c for poly in u for c in poly] + list(v)
        self.assertFalse(all(c == ord("A") for c in flat))

    def test_10_wrong_sk_gives_garbage(self):
        msg       = os.urandom(CHUNK)
        ct        = self.eng.encrypt(self.pk, msg)
        _, bad_sk = self.eng.keygen()          # different key pair
        rec       = self.eng.decrypt(bad_sk, ct)
        self.assertNotEqual(rec, msg)          # decryption should fail silently


# ─────────────────────────────────────────────────────────────────────────────
# 2. String payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestStringPayloads(unittest.TestCase):

    def setUp(self):
        Profiler.current_category = "Strings"
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def _rt(self, text):
        return _roundtrip_str(self.eng, self.pk, self.sk, text)

    def test_11_hello_world(self):
        self.assertTrue(self._rt("Hello, World!"))

    def test_12_single_char(self):
        self.assertTrue(self._rt("X"))

    def test_13_exactly_32_chars(self):
        self.assertTrue(self._rt("A" * 32))

    def test_14_exactly_33_chars_triggers_two_chunks(self):
        text = "B" * 33
        cts  = self.eng.encrypt_chunked(self.pk, text)
        self.assertEqual(len(cts), 2)
        self.assertEqual(
            self.eng.decrypt_chunked(self.sk, cts, output_type="str"), text
        )

    def test_15_long_string_190_chars(self):
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
            " 1234567890!"
        )
        self.assertTrue(self._rt(text))

    def test_16_unicode_emoji(self):
        self.assertTrue(self._rt("Kyber 🔐🛡️ post-quantum!"))

    def test_17_unicode_cjk(self):
        self.assertTrue(self._rt("量子暗号は未来のセキュリティです。"))

    def test_18_numeric_string(self):
        self.assertTrue(self._rt("3141592653589793238462643383279502884197"))

    def test_19_whitespace_only(self):
        self.assertTrue(self._rt("   \t\n   "))

    def test_20_repeated_pattern(self):
        self.assertTrue(self._rt("kyber" * 20))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Integer payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegerPayloads(unittest.TestCase):

    def setUp(self):
        Profiler.current_category = "Integers"
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def _rt(self, val):
        return _roundtrip_int(self.eng, self.pk, self.sk, val)

    def test_21_zero(self):
        self.assertTrue(self._rt(0))

    def test_22_one(self):
        self.assertTrue(self._rt(1))

    def test_23_small_int(self):
        self.assertTrue(self._rt(42))

    def test_24_medium_int(self):
        self.assertTrue(self._rt(123456789))

    def test_25_large_int(self):
        self.assertTrue(self._rt(2**128 - 1))

    def test_26_very_large_int(self):
        self.assertTrue(self._rt(2**256 - 1))

    def test_27_prime(self):
        self.assertTrue(self._rt(3329))    # Kyber's own modulus

    def test_28_power_of_two(self):
        self.assertTrue(self._rt(2**64))


# ─────────────────────────────────────────────────────────────────────────────
# 4. Float payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestFloatPayloads(unittest.TestCase):

    def setUp(self):
        Profiler.current_category = "Floats"
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def _rt(self, val):
        return _roundtrip_float(self.eng, self.pk, self.sk, val)

    def test_29_pi(self):
        import math
        self.assertTrue(self._rt(math.pi))

    def test_30_negative_float(self):
        self.assertTrue(self._rt(-2.718281828))

    def test_31_zero_float(self):
        self.assertTrue(self._rt(0.0))

    def test_32_very_small_float(self):
        self.assertTrue(self._rt(1e-300))

    def test_33_infinity(self):
        import math
        self.assertTrue(self._rt(math.inf))

    def test_34_nan(self):
        import math
        self.assertTrue(self._rt(math.nan))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bytes payloads
# ─────────────────────────────────────────────────────────────────────────────

class TestBytesPayloads(unittest.TestCase):

    def setUp(self):
        Profiler.current_category = "Bytes"
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def _rt(self, data):
        return _roundtrip_bytes(self.eng, self.pk, self.sk, data)

    def test_35_random_32_bytes(self):
        self.assertTrue(self._rt(os.urandom(32)))

    def test_36_random_100_bytes(self):
        self.assertTrue(self._rt(os.urandom(100)))

    def test_37_all_zeros(self):
        self.assertTrue(self._rt(bytes(32)))

    def test_38_all_ones(self):
        self.assertTrue(self._rt(b"\xff" * 32))

    def test_39_structured_bytes(self):
        data = bytes(range(256))
        self.assertTrue(self._rt(data))

    def test_40_single_byte(self):
        self.assertTrue(self._rt(b"\xab"))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Chunked API
# ─────────────────────────────────────────────────────────────────────────────

class TestChunkedAPI(unittest.TestCase):

    def setUp(self):
        # Profiler.current_category = "Chunked"
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    def test_41_chunk_count_exact_multiple(self):
        # Subtracting 5 bytes accounts for the header overhead 
        # so the final payload aligns perfectly with the 3-chunk boundary.
        data = os.urandom(CHUNK * 3 - 5)
        cts  = self.eng.encrypt_chunked(self.pk, data)
        self.assertEqual(len(cts), 3)

    def test_42_chunk_count_with_remainder(self):
        data = os.urandom(CHUNK * 2 + 5)
        cts  = self.eng.encrypt_chunked(self.pk, data)
        self.assertEqual(len(cts), 3)

    def test_43_each_chunk_independently_valid(self):
        text = "Hello Kyber chunked world! " * 4
        cts  = self.eng.encrypt_chunked(self.pk, text)
        for ct in cts:
            raw = self.eng.decrypt(self.sk, ct)
            self.assertEqual(len(raw), CHUNK)

    def test_44_prepare_payload_string(self):
        chunks = self.eng.prepare_payload("test")
        self.assertTrue(all(len(c) == CHUNK for c in chunks))

    def test_45_prepare_payload_int(self):
        chunks = self.eng.prepare_payload(999)
        self.assertTrue(all(len(c) == CHUNK for c in chunks))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Compression
# ─────────────────────────────────────────────────────────────────────────────

class TestCompression(unittest.TestCase):

    def setUp(self):
        Profiler.current_category = "Compression"
        self.raw_eng  = make_engine(k=2, compress=False)
        self.comp_eng = make_engine(k=2, compress=True)
        self.pk, self.sk = self.raw_eng.keygen()

    def test_46_compressed_ciphertext_is_bytes(self):
        msg = os.urandom(CHUNK)
        ct  = self.comp_eng.encrypt(self.pk, msg)
        self.assertIsInstance(ct, bytes)

    def test_47_compressed_roundtrip(self):
        msg = os.urandom(CHUNK)
        ct  = self.comp_eng.encrypt(self.pk, msg)
        rec = self.comp_eng.decrypt(self.sk, ct)
        # Compression is lossy at the coefficient level but decoding still
        # recovers the original message bits (noise stays within q/4).
        self.assertEqual(rec, msg)

    def test_48_compression_ratio_below_one(self):
        msg  = os.urandom(CHUNK)
        ct   = self.raw_eng.encrypt(self.pk, msg)
        comp = CiphertextCompressor(q=Q)
        info = comp.compression_ratio(ct)
        self.assertLess(info["ratio"], 1.0)
        self.assertGreater(info["savings_pct"], 0)

    def test_49_magic_header_present(self):
        msg  = os.urandom(CHUNK)
        blob = self.comp_eng.encrypt(self.pk, msg)
        self.assertTrue(blob[:4] == b"KYBR")

    def test_50_tampered_blob_raises(self):
        msg  = os.urandom(CHUNK)
        blob = bytearray(self.comp_eng.encrypt(self.pk, msg))
        blob[0] = 0xFF    # corrupt magic
        with self.assertRaises((DecompressionError, InvalidCiphertextError)):
            self.comp_eng.decrypt(self.sk, bytes(blob))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling(unittest.TestCase):

    def setUp(self):
        self.eng = make_engine(k=2)
        self.pk, self.sk = self.eng.keygen()

    # keygen errors
    def test_err_01_bad_seed_type(self):
        with self.assertRaises(InvalidSeedError):
            self.eng.keygen(seed=12345)

    def test_err_02_seed_wrong_length(self):
        with self.assertRaises(InvalidSeedError):
            self.eng.keygen(seed=b"tooshort")

    def test_err_03_invalid_parameter_n(self):
        with self.assertRaises(ParameterError):
            KyberCoreEngine(n=300, q=Q, eta=ETA, k_dimension=2)   # 300 not power-of-2

    def test_err_04_invalid_parameter_k(self):
        with self.assertRaises(ParameterError):
            KyberCoreEngine(n=N, q=Q, eta=ETA, k_dimension=0)

    # encrypt errors
    def test_err_05_bad_public_key_type(self):
        with self.assertRaises(InvalidPublicKeyError):
            self.eng.encrypt("not_a_key", os.urandom(CHUNK))

    def test_err_06_bad_public_key_rho(self):
        bad_pk = (b"short", self.pk[1])
        with self.assertRaises(InvalidPublicKeyError):
            self.eng.encrypt(bad_pk, os.urandom(CHUNK))

    def test_err_07_bad_public_key_t_length(self):
        bad_pk = (self.pk[0], [self.pk[1][0]])   # t has only 1 poly instead of 2
        with self.assertRaises(InvalidPublicKeyError):
            self.eng.encrypt(bad_pk, os.urandom(CHUNK))

    def test_err_08_message_wrong_size(self):
        with self.assertRaises(MessageSizeError):
            self.eng.encrypt(self.pk, b"tooshort")

    def test_err_09_message_too_long(self):
        with self.assertRaises(MessageSizeError):
            self.eng.encrypt(self.pk, os.urandom(CHUNK + 1))

    def test_err_10_message_wrong_type(self):
        with self.assertRaises(InvalidMessageTypeError):
            self.eng.encrypt(self.pk, 12345)   # int, not bytes

    def test_err_11_empty_message(self):
        with self.assertRaises(EmptyMessageError):
            self.eng.encrypt(self.pk, b"")

    def test_err_12_encrypt_chunked_empty_string(self):
        # empty string → _to_bytes returns b"" → EmptyMessageError
        with self.assertRaises(EmptyMessageError):
            self.eng.encrypt_chunked(self.pk, "")

    # decrypt errors
    def test_err_13_bad_secret_key_type(self):
        ct = self.eng.encrypt(self.pk, os.urandom(CHUNK))
        with self.assertRaises(InvalidSecretKeyError):
            self.eng.decrypt("bad_sk", ct)

    def test_err_14_bad_secret_key_length(self):
        ct = self.eng.encrypt(self.pk, os.urandom(CHUNK))
        with self.assertRaises(InvalidSecretKeyError):
            self.eng.decrypt([self.sk[0]], ct)    # only 1 poly instead of 2

    def test_err_15_malformed_ciphertext_tuple(self):
        with self.assertRaises(InvalidCiphertextError):
            self.eng.decrypt(self.sk, ([], [], []))   # 3-tuple, not 2

    def test_err_16_ciphertext_u_wrong_k(self):
        ct = self.eng.encrypt(self.pk, os.urandom(CHUNK))
        u, v = ct
        bad_ct = ([u[0]], v)   # u has 1 poly instead of k=2
        with self.assertRaises(InvalidCiphertextError):
            self.eng.decrypt(self.sk, bad_ct)

    def test_err_17_decrypt_chunked_non_list(self):
        ct = self.eng.encrypt(self.pk, os.urandom(CHUNK))
        with self.assertRaises(CiphertextListError):
            self.eng.decrypt_chunked(self.sk, ct)   # should be a list


# ─────────────────────────────────────────────────────────────────────────────
# 9. Multi-variant
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiVariant(unittest.TestCase):

    def _run_variant(self, k: int, label: str):
        eng    = make_engine(k=k)
        pk, sk = eng.keygen()
        text   = f"Kyber-{label} variant test payload — index {k}."
        cts    = eng.encrypt_chunked(pk, text)
        result = eng.decrypt_chunked(sk, cts, output_type="str")
        self.assertEqual(result, text, f"{label} round-trip failed")

    def test_mv_01_kyber512(self):
        self._run_variant(k=2, label="512")

    def test_mv_02_kyber768(self):
        self._run_variant(k=3, label="768")

    def test_mv_03_kyber1024(self):
        self._run_variant(k=4, label="1024")

    def test_mv_04_cross_key_mismatch_512_768(self):
        """A secret key from k=2 must not work on a k=3 ciphertext."""
        eng2 = make_engine(k=2)
        eng3 = make_engine(k=3)
        pk3, _   = eng3.keygen()
        _, sk2   = eng2.keygen()
        msg      = os.urandom(CHUNK)
        ct3      = eng3.encrypt(pk3, msg)
        # Decryption with wrong sk dimension must raise, not silently corrupt
        with self.assertRaises(Exception):
            eng3.decrypt(sk2, ct3)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Noise analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseAnalysis(unittest.TestCase):

    def _run_noise(self, k: int) -> dict:
        eng    = make_engine(k=k)
        pk, sk = eng.keygen()
        msg    = os.urandom(CHUNK)
        ct     = eng.encrypt(pk, msg)
        u, v   = ct

        # Compute w = v - s^T u
        sTu = eng.vmm.vec_dot(sk, u)
        w   = eng.alu.poly_sub(v, sTu)

        # Encode original message to polynomial
        ser  = SerializationUnit(N, Q)
        mbar = ser.encode_message_to_poly(msg)

        # Noise = w - mbar (centered)
        noise = [(w[i] - mbar[i]) % Q for i in range(N)]
        noise_c = [eng.alu.centered_mod(n) for n in noise]

        return {
            "max_abs":   max(abs(n) for n in noise_c),
            "mean_abs":  sum(abs(n) for n in noise_c) / N,
            "bound":     Q // 4,
            "all_ok":    all(abs(n) < Q // 4 for n in noise_c),
            "noise":     noise_c,
        }

    def test_noise_01_kyber512_within_bound(self):
        stats = self._run_noise(k=2)
        self.assertTrue(
            stats["all_ok"],
            f"Noise exceeded q/4={stats['bound']}; max={stats['max_abs']}"
        )

    def test_noise_02_kyber768_within_bound(self):
        stats = self._run_noise(k=3)
        self.assertTrue(stats["all_ok"])

    def test_noise_03_kyber1024_within_bound(self):
        stats = self._run_noise(k=4)
        self.assertTrue(stats["all_ok"])

    def test_noise_04_max_well_below_bound(self):
        # Empirically, max noise should be << q/4 (≈ 832); typically < 200
        stats = self._run_noise(k=2)
        self.assertLess(stats["max_abs"], Q // 4)

    def test_noise_05_mean_small(self):
        stats = self._run_noise(k=2)
        self.assertLess(stats["mean_abs"], 200)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Benchmarks (run last; results printed by the runner)
# ─────────────────────────────────────────────────────────────────────────────

class TestBenchmarks(unittest.TestCase):
    """
    These tests always pass — they exist to collect timing data.
    Results are printed by the custom runner at the end.
    """

    def _bench_variant(self, k: int, label: str):
        eng = make_engine(k=k)

        # KeyGen
        t0 = time.perf_counter()
        for _ in range(5):
            pk, sk = eng.keygen()
        keygen_ms = (time.perf_counter() - t0) * 1000 / 5
        _bench_log.append((f"KeyGen   Kyber-{label}", keygen_ms))

        # Encrypt single chunk
        msg = os.urandom(CHUNK)
        t0  = time.perf_counter()
        for _ in range(5):
            ct = eng.encrypt(pk, msg)
        enc_ms = (time.perf_counter() - t0) * 1000 / 5
        _bench_log.append((f"Encrypt  Kyber-{label} (1 chunk)", enc_ms))

        # Decrypt single chunk
        t0 = time.perf_counter()
        for _ in range(5):
            eng.decrypt(sk, ct)
        dec_ms = (time.perf_counter() - t0) * 1000 / 5
        _bench_log.append((f"Decrypt  Kyber-{label} (1 chunk)", dec_ms))

        # Chunked round-trip (190-char string → 6 chunks)
        long_msg = "The quick brown fox jumps over the lazy dog. " * 4
        t0 = time.perf_counter()
        for _ in range(3):
            cts = eng.encrypt_chunked(pk, long_msg)
            eng.decrypt_chunked(sk, cts, output_type="str")
        rt_ms = (time.perf_counter() - t0) * 1000 / 3
        _bench_log.append((f"RoundTrip Kyber-{label} (6 chunks)", rt_ms))

    def _bench_compression(self):
        eng  = make_engine(k=2, compress=True)
        pk, sk = eng.keygen()
        msg  = os.urandom(CHUNK)

        t0 = time.perf_counter()
        for _ in range(5):
            blob = eng.encrypt(pk, msg)
        comp_enc_ms = (time.perf_counter() - t0) * 1000 / 5
        _bench_log.append(("Compress-Encrypt Kyber-512", comp_enc_ms))

        t0 = time.perf_counter()
        for _ in range(5):
            eng.decrypt(sk, blob)
        comp_dec_ms = (time.perf_counter() - t0) * 1000 / 5
        _bench_log.append(("Compress-Decrypt Kyber-512", comp_dec_ms))

        # Report compression ratio
        raw_eng = make_engine(k=2)
        ct_raw  = raw_eng.encrypt(pk, msg)
        comp    = CiphertextCompressor(q=Q)
        info    = comp.compression_ratio(ct_raw)
        _bench_log.append(
            (f"Compression ratio (d_u=10, d_v=4)",
             float(f"{info['savings_pct']:.2f}"))   # store savings_pct as "ms" column
        )

    def test_bench_01_kyber512(self):
        self._bench_variant(k=2, label="512")

    def test_bench_02_kyber768(self):
        self._bench_variant(k=3, label="768")

    def test_bench_03_kyber1024(self):
        self._bench_variant(k=4, label="1024")

    def test_bench_04_compression(self):
        self._bench_compression()


# ─────────────────────────────────────────────────────────────────────────────
# Custom runner that prints the benchmark report after all tests
# ─────────────────────────────────────────────────────────────────────────────

from ..tests.helpers import print_bench_report


def _run() -> None:
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()

    test_classes = [
        TestCoreCorrectness,
        TestStringPayloads,
        TestIntegerPayloads,
        TestFloatPayloads,
        TestBytesPayloads,
        TestChunkedAPI,
        TestCompression,
        TestErrorHandling,
        TestMultiVariant,
        TestNoiseAnalysis,
        TestBenchmarks,
    ]

    labels = [
        "Core Correctness",
        "String Payloads",
        "Integer Payloads",
        "Float Payloads",
        "Bytes Payloads",
        "Chunked API",
        "Compression",
        "Error Handling",
        "Multi-Variant",
        "Noise Analysis",
        "Benchmarks",
    ]

    for cls, label in zip(test_classes, labels):
        section(label)
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=__import__("sys").stdout)
    result = runner.run(suite)

    print_bench_report()
    Profiler.generate_html_dashboard()

    raise SystemExit(0 if result.wasSuccessful() else 1) 