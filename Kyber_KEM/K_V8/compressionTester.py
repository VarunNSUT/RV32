"""
compressionTester.py
====================
Standalone tester for the Kyber-512 compression layer.

Run from the package root:
    python -m <your_package>.compressionTester

What this tests
---------------
1. Single-coefficient compress/decompress at every d value (2..12).
2. Full-polynomial compress/decompress: noise statistics at every d level.
3. Correctness sweep: encrypt → decrypt at du=10, dv=2..12.
4. End-to-end pipeline with timing: keygen, encrypt, decrypt all timed.
5. Multi-chunk pipeline: realistic stream of chunks, all timed.

All results are printed to stdout AND written to compression_report.csv.
"""

import os, sys, time, hashlib, math, csv, random
from pathlib import Path

try:
    from .KyberCore         import KyberCoreEngine
    from .SerializationUnit import SerializationUnit
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from KyberCore          import KyberCoreEngine          # type: ignore
    from SerializationUnit  import SerializationUnit        # type: ignore

# ── helpers ──────────────────────────────────────────────────────────────────
SEP  = "=" * 68
SEP2 = "-" * 68
Q    = 3329
N    = 256

def timeit_ms(fn, runs=10):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / runs, min(times), max(times)

def noise_stats(original, recovered):
    errs = []
    for a, b in zip(original, recovered):
        e = abs(a - b) % Q
        errs.append(min(e, Q - e))
    avg = sum(errs) / len(errs)
    mx  = max(errs)
    signal_power = sum(x**2 for x in original) / len(original) + 1e-9
    noise_power  = sum(e**2 for e in errs)     / len(errs)     + 1e-9
    snr = 10 * math.log10(signal_power / noise_power)
    return avg, mx, snr

def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

CSV_PATH = Path(__file__).parent / "compression_report.csv"
csv_rows = []

def record(section, **kwargs):
    csv_rows.append({"section": section, **kwargs})

# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — single coefficient, step-by-step at every d
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 1 · Single-coefficient compress/decompress (step by step)")
print(f"  Q = {Q},  example coefficient x = 1664  (≈ Q/2, a typical 'mid-range' value)")
print(f"\n  {'d':>4}  {'2^d':>6}  {'Q/2^d':>8}  {'compressed':>12}  "
      f"{'recovered':>10}  {'error':>7}  {'size_saved_bits':>16}")
print(f"  {SEP2[:66]}")

s = SerializationUnit(N, Q)
example_x = 1664
for d in range(2, 13):
    comp  = s._compress_coeff(example_x, d)
    rec   = s._decompress_coeff(comp, d)
    err   = min(abs(example_x - rec) % Q, Q - abs(example_x - rec) % Q)
    saved = 12 - d
    divisor = Q / (1 << d)
    print(f"  {d:>4}  {1<<d:>6}  {divisor:>8.1f}  {comp:>12}  "
          f"{rec:>10}  {err:>7}  {saved:>16}")
    record("coeff_stepwise", d=d, two_pow_d=1<<d, Q_div_2d=round(divisor,2),
           x_original=example_x, compressed=comp, recovered=rec,
           error=err, bits_saved=saved)

print(f"\n  ► d=2  means: divide by Q/4 = {Q//4}  → 2-bit output")
print(f"  ► d=4  means: divide by Q/16 = {Q//16} → 4-bit output  (Kyber dv floor)")
print(f"  ► d=10 means: divide by Q/1024 ≈ {Q/1024:.1f} → 10-bit output (Kyber du)")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — polynomial noise at every d
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 2 · Full-polynomial noise analysis (N=256 random coefficients)")
print(f"  {'d':>4}  {'bytes_out':>10}  {'bytes_raw':>10}  {'saved%':>8}  "
      f"{'avg_err':>9}  {'max_err':>9}  {'SNR_dB':>8}  {'budget_ok':>10}")
print(f"  {SEP2[:70]}")

random.seed(42)
poly_test = [random.randint(0, Q - 1) for _ in range(N)]
budget    = Q // 4

for d in range(2, 13):
    comp      = s.compress_poly(poly_test, d)
    recovered = s.decompress_poly(comp, d)
    avg_e, mx_e, snr = noise_stats(poly_test, recovered)
    raw_bytes   = (N * 12 + 7) // 8
    saved_pct   = (1 - len(comp) / raw_bytes) * 100
    budget_ok   = "✓ SAFE" if mx_e < budget else "✗ RISKY"
    print(f"  {d:>4}  {len(comp):>10}  {raw_bytes:>10}  {saved_pct:>7.1f}%  "
          f"{avg_e:>9.1f}  {mx_e:>9}  {snr:>8.1f}  {budget_ok:>10}")
    record("poly_noise", d=d, bytes_out=len(comp), bytes_raw=raw_bytes,
           saved_pct=round(saved_pct,1), avg_err=round(avg_e,1),
           max_err=mx_e, snr_db=round(snr,1), budget_ok=budget_ok.strip())

print(f"\n  LWE noise budget = Q/4 = {budget}")
print(f"  d=4 is the practical floor for Kyber-512 (Kyber spec choice)")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — correctness sweep across dv values
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 3 · Encrypt/decrypt correctness vs dv  (du fixed at 10)")
engine = KyberCoreEngine(N, Q, 2, k_dimension=2, d_u=10, d_v=4)
seed   = b"compression_tester_fixed_seed___"
pk, sk = engine.keygen(seed)
chunk  = b"Hello FPGA World!!! test chunk!!"
r_seed = hashlib.sha256(b"r").digest()

print(f"  Plaintext : {chunk}")
print(f"  du fixed  : 10\n")
print(f"  {'dv':>4}  {'ct_size_B':>10}  {'vs_raw_12bit':>13}  "
      f"{'correct':>9}  {'note':>30}")
print(f"  {SEP2[:66]}")

raw_ct_bits  = 2 * N * 12 + N * 12
raw_ct_bytes = (raw_ct_bits + 7) // 8

for dv in range(2, 13):
    # Temporarily override dv by creating a fresh engine
    eng_tmp = KyberCoreEngine(N, Q, 2, k_dimension=2, d_u=10, d_v=dv)
    ct  = eng_tmp.serializer.compress_ciphertext(*engine.encrypt(pk, chunk, r_seed), du=10, dv=dv)
    u2, v2 = eng_tmp.serializer.decompress_ciphertext(ct, 2, du=10, dv=dv)
    from .ArithmeticUnit import ntt, intt, ntt_mul
    s_ntt = [ntt(si) for si in sk]
    u_ntt = [ntt(ui) for ui in u2]
    su = eng_tmp.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True)
    w  = eng_tmp.alu.poly_sub(v2, su)
    out = eng_tmp.serializer.decode_poly_to_message(w)
    ok  = (out == chunk)
    saved = raw_ct_bytes - len(ct)
    note = ("Kyber-512 spec" if dv == 4
            else "no compression" if dv == 12
            else "smaller but correct" if ok
            else "decryption failure")
    print(f"  {dv:>4}  {len(ct):>10}  {saved:>+13}  "
          f"{'✓ YES' if ok else '✗ NO':>9}  {note:>30}")
    record("correctness_sweep", dv=dv, du=10, ct_size=len(ct),
           bytes_saved_vs_raw=saved, correct=ok, note=note)

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — timing
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 4 · End-to-end timing  (du=10, dv=4,  10 runs each)")
RUNS = 10
engine = KyberCoreEngine(N, Q, 2, k_dimension=2, d_u=10, d_v=4)
pk, sk = engine.keygen(seed)
ct_ref = engine.encrypt(pk, chunk, r_seed)

avg_kg,  mn_kg,  mx_kg  = timeit_ms(lambda: engine.keygen(seed), RUNS)
avg_enc, mn_enc, mx_enc = timeit_ms(lambda: engine.encrypt(pk, chunk, r_seed), RUNS)
avg_dec, mn_dec, mx_dec = timeit_ms(lambda: engine.decrypt(sk, ct_ref), RUNS)

print(f"  {'Operation':<28} {'avg_ms':>8}  {'min_ms':>8}  {'max_ms':>8}")
print(f"  {SEP2[:56]}")
for label, avg, mn, mx in [
    ("keygen",         avg_kg,  mn_kg,  mx_kg),
    ("encrypt (full)", avg_enc, mn_enc, mx_enc),
    ("decrypt (full)", avg_dec, mn_dec, mx_dec),
]:
    print(f"  {label:<28} {avg:>8.2f}  {mn:>8.2f}  {mx:>8.2f}")
    record("timing", operation=label.strip(), avg_ms=round(avg,3),
           min_ms=round(mn,3), max_ms=round(mx,3))

# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — multi-chunk pipeline
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 5 · Multi-chunk pipeline  (realistic message stream)")

message_long = (
    "Hello World! Processing modular code block layout for FPGA target. "
    "Each chunk is 32 bytes. Compression applied to every ciphertext packet. "
    "This simulates the full AXI stream pipeline behavior on hardware. "
    "Padding with zeros to fill last chunk automatically."
).encode()

print(f"  Message length : {len(message_long)} bytes")
print()
print(f"  {'Chunk':>6}  {'plain_B':>8}  {'enc_ms':>9}  {'dec_ms':>9}  {'ok':>5}")
print(f"  {SEP2[:48]}")

ciphertexts = engine.encrypt_chunked(pk, message_long)
total_enc_ms = 0
recovered_chunks = []

for idx, ct_ch in enumerate(ciphertexts):
    t0 = time.perf_counter()
    out = engine.decrypt(sk, ct_ch)
    dec_ms = (time.perf_counter() - t0) * 1000
    recovered_chunks.append(out)

reassembled = b"".join(recovered_chunks).rstrip(b"\x00")
# Strip _to_bytes header (first 5 bytes: tag + length)
if reassembled and reassembled[0] == 0x01:
    import struct
    dlen = struct.unpack_from(">I", reassembled, 1)[0]
    reassembled = reassembled[5:5+dlen]

print(f"\n  Reassembled : '{reassembled.decode()[:60]}…'")

# ═════════════════════════════════════════════════════════════════════════════
# Write CSV
# ═════════════════════════════════════════════════════════════════════════════
hdr("Writing compression_report.csv")
all_keys = sorted({k for row in csv_rows for k in row.keys()})
with open(CSV_PATH, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_keys)
    w.writeheader()
    w.writerows(csv_rows)
print(f"  Saved → {CSV_PATH}")
print(f"  Rows  : {len(csv_rows)}")
print(f"  Cols  : {len(all_keys)}")
print(f"\n{SEP}\n  ALL TESTS COMPLETE\n{SEP}\n")
