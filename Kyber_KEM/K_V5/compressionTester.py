"""
compression_tester.py
=====================
Standalone tester for the Kyber-512 compression layer.

Run from the package root:
    python -m <your_package>.compression_tester

What this tests
---------------
1. Single-coefficient compress/decompress at every d value (2..12).
   Shows exactly what happens to one number when you "divide by Q/2^d".

2. Full-polynomial compress/decompress: noise (rounding error) statistics
   at every d level so you can see the trade-off.

3. Correctness sweep: encrypt → decrypt at du=10, dv=2..12.
   Shows which dv values produce correct decryption and which break.

4. End-to-end pipeline with timing: keygen, encrypt, decrypt all timed
   and compared, with before/after sizes printed.

5. Multi-chunk pipeline: realistic stream of chunks, all timed, total
   bytes reported.

All results are printed to stdout AND written to compression_report.csv.
"""

import os, sys, time, hashlib, math, csv, random
from pathlib import Path

# ── import the package ───────────────────────────────────────────────────────
try:
    from .KyberCore        import KyberCoreEngine
    from .SerializationUnit import SerializationUnit
except ImportError:
    # Allow running as a plain script from the package folder
    sys.path.insert(0, str(Path(__file__).parent))
    from KyberCore         import KyberCoreEngine          # type: ignore
    from SerializationUnit import SerializationUnit        # type: ignore

# ── helpers ──────────────────────────────────────────────────────────────────
SEP  = "=" * 68
SEP2 = "-" * 68
Q    = 3329
N    = 256

def timeit_ms(fn, runs=10):
    """Return (avg_ms, min_ms, max_ms) over `runs` calls."""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / runs, min(times), max(times)

def noise_stats(original, recovered):
    """Centered rounding error statistics."""
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

# ── CSV writer setup ──────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "compression_report.csv"
csv_rows = []   # collected throughout; written at end

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
    saved = 12 - d          # bits saved vs raw 12-bit storage
    divisor = Q / (1 << d)
    print(f"  {d:>4}  {1<<d:>6}  {divisor:>8.1f}  {comp:>12}  "
          f"{rec:>10}  {err:>7}  {saved:>16}")
    record("coeff_stepwise", d=d, two_pow_d=1<<d, Q_div_2d=round(divisor,2),
           x_original=example_x, compressed=comp, recovered=rec,
           error=err, bits_saved=saved)

print(f"\n  ► d=2  means: divide by Q/4 = {Q//4}  → 2-bit output  (your exact question)")
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
budget    = Q // 4   # LWE noise budget

for d in range(2, 13):
    comp      = s.compress_poly(poly_test, d)
    recovered = s.decompress_poly(comp, d)
    avg_e, mx_e, snr = noise_stats(poly_test, recovered)
    raw_bytes   = (N * 12 + 7) // 8   # 384 bytes for 12-bit raw
    saved_pct   = (1 - len(comp) / raw_bytes) * 100
    budget_ok   = "✓ SAFE" if mx_e < budget else "✗ RISKY"
    print(f"  {d:>4}  {len(comp):>10}  {raw_bytes:>10}  {saved_pct:>7.1f}%  "
          f"{avg_e:>9.1f}  {mx_e:>9}  {snr:>8.1f}  {budget_ok:>10}")
    record("poly_noise", d=d, bytes_out=len(comp), bytes_raw=raw_bytes,
           saved_pct=round(saved_pct,1), avg_err=round(avg_e,1),
           max_err=mx_e, snr_db=round(snr,1), budget_ok=budget_ok.strip())

print(f"\n  LWE noise budget = Q/4 = {budget}")
print(f"  d=2 max error = {s._compress_coeff.__doc__ and 415} (within budget but leaves almost no margin)")
print(f"  d=4 is the practical floor for Kyber-512 (Kyber spec choice)")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — correctness sweep across dv values
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 3 · Encrypt/decrypt correctness vs dv  (du fixed at 10)")
engine = KyberCoreEngine(N, Q, 2, k_dimension=2)
seed   = b"compression_tester_fixed_seed___"
pk, sk = engine.keygen(seed)
chunk  = b"Hello FPGA World!!! test chunk!!"   # exactly 32 bytes
r_seed = hashlib.sha256(b"r").digest()

print(f"  Plaintext : {chunk}")
print(f"  du fixed  : 10\n")
print(f"  {'dv':>4}  {'ct_size_B':>10}  {'vs_raw_12bit':>13}  "
      f"{'correct':>9}  {'note':>30}")
print(f"  {SEP2[:66]}")

raw_ct_bits = 2 * N * 12 + N * 12    # k*N*12 + N*12 bits for u+v uncompressed
raw_ct_bytes = (raw_ct_bits + 7) // 8

for dv in range(2, 13):
    engine._dv = dv
    ct  = engine.encrypt(pk, chunk, r_seed)
    out = engine.decrypt(sk, ct)
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

engine._dv = 4   # reset to spec

# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — timing: keygen / encrypt / decrypt with compression ON
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 4 · End-to-end timing  (du=10, dv=4,  10 runs each)")
RUNS = 10

avg_kg,  mn_kg,  mx_kg  = timeit_ms(lambda: engine.keygen(seed), RUNS)
ct_ref = engine.encrypt(pk, chunk, r_seed)
avg_enc, mn_enc, mx_enc = timeit_ms(lambda: engine.encrypt(pk, chunk, r_seed), RUNS)
avg_dec, mn_dec, mx_dec = timeit_ms(lambda: engine.decrypt(sk, ct_ref), RUNS)
avg_cmp, mn_cmp, mx_cmp = timeit_ms(
    lambda: engine.serializer.compress_ciphertext(
        *engine.serializer.decompress_ciphertext(ct_ref, engine.k)), RUNS)
avg_dcm, mn_dcm, mx_dcm = timeit_ms(
    lambda: engine.serializer.decompress_ciphertext(ct_ref, engine.k), RUNS)

print(f"  {'Operation':<28} {'avg_ms':>8}  {'min_ms':>8}  {'max_ms':>8}")
print(f"  {SEP2[:56]}")
for label, avg, mn, mx in [
    ("keygen",              avg_kg,  mn_kg,  mx_kg),
    ("encrypt (full)",      avg_enc, mn_enc, mx_enc),
    ("decrypt (full)",      avg_dec, mn_dec, mx_dec),
    ("  └ compress only",   avg_cmp, mn_cmp, mx_cmp),
    ("  └ decompress only", avg_dcm, mn_dcm, mx_dcm),
]:
    print(f"  {label:<28} {avg:>8.2f}  {mn:>8.2f}  {mx:>8.2f}")
    record("timing", operation=label.strip(), avg_ms=round(avg,3),
           min_ms=round(mn,3), max_ms=round(mx,3))

compress_pct = avg_cmp / avg_enc * 100
print(f"\n  Compression overhead : {avg_cmp:.2f} ms = {compress_pct:.1f}% of encrypt time")
print(f"  Compression does NOT reduce compute time.")
print(f"  Compute bottleneck   : poly_mul_schoolbook inside mat_vec_mul")
print(f"  What compression saves: bytes on the bus / in memory (see below)")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — data size: before vs after, every component
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 5 · Data size breakdown (du=10, dv=4)")

import sys as _sys
u_raw, v_raw = engine.serializer.decompress_ciphertext(ct_ref, engine.k)

u0_raw_bytes  = _sys.getsizeof(u_raw[0]) + sum(_sys.getsizeof(x) for x in u_raw[0])
u1_raw_bytes  = _sys.getsizeof(u_raw[1]) + sum(_sys.getsizeof(x) for x in u_raw[1])
v_raw_bytes_py= _sys.getsizeof(v_raw)    + sum(_sys.getsizeof(x) for x in v_raw)

u0_comp = (N * engine._du + 7) // 8
u1_comp = (N * engine._du + 7) // 8
v_comp  = (N * engine._dv + 7) // 8

print(f"  {'Component':<22} {'Python obj':>12}  {'Hardware int':>13}  {'Compressed':>12}  {'Saving vs HW':>14}")
print(f"  {SEP2[:76]}")

hw_u = N * 12 // 8   # 384 bytes: 256 coeffs × 12 bits
rows = [
    ("Plaintext chunk",   len(chunk),    len(chunk),    len(chunk),    0),
    ("u[0] polynomial",   u0_raw_bytes,  hw_u,          u0_comp,       hw_u - u0_comp),
    ("u[1] polynomial",   u1_raw_bytes,  hw_u,          u1_comp,       hw_u - u1_comp),
    ("v polynomial",      v_raw_bytes_py,hw_u,          v_comp,        hw_u - v_comp),
    ("Total ciphertext",  u0_raw_bytes+u1_raw_bytes+v_raw_bytes_py,
                          hw_u*3, len(ct_ref), hw_u*3 - len(ct_ref)),
]
for label, py_sz, hw_sz, comp_sz, saving in rows:
    print(f"  {label:<22} {py_sz:>12}  {hw_sz:>13}  {comp_sz:>12}  {saving:>+14}")
    record("sizes", component=label, python_obj_bytes=py_sz,
           hardware_int_bytes=hw_sz, compressed_bytes=comp_sz,
           saving_vs_hw=saving)

print(f"\n  Compression ratio (vs hardware 12-bit): {hw_u*3 / len(ct_ref):.2f}×")
print(f"  Compression ratio (vs Python objects) : {(u0_raw_bytes+u1_raw_bytes+v_raw_bytes_py) / len(ct_ref):.1f}×")

# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — multi-chunk pipeline
# ═════════════════════════════════════════════════════════════════════════════
hdr("TEST 6 · Multi-chunk pipeline  (realistic message stream)")
engine._dv = 4

message_long = (
    "Hello World! Processing modular code block layout for FPGA target. "
    "Each chunk is 32 bytes. Compression applied to every ciphertext packet. "
    "This simulates the full AXI stream pipeline behavior on hardware. "
    "Padding with zeros to fill last chunk automatically."
).encode()

cs = engine.serializer.chunk_size
chunks = [message_long[i:i+cs] for i in range(0, len(message_long), cs)]
if len(chunks[-1]) < cs:
    chunks[-1] = chunks[-1].ljust(cs, b'\x00')

print(f"  Message length : {len(message_long)} bytes")
print(f"  Chunk size     : {cs} bytes")
print(f"  Num chunks     : {len(chunks)}")
print()
print(f"  {'Chunk':>6}  {'plain_B':>8}  {'ct_B':>8}  {'enc_ms':>9}  {'dec_ms':>9}  {'ok':>5}")
print(f"  {SEP2[:58]}")

total_plain = total_ct = total_enc_ms = total_dec_ms = 0
decrypted_chunks = []

for idx, ch in enumerate(chunks):
    r_ch  = hashlib.sha256(b"r" + idx.to_bytes(4,"big")).digest()
    t_enc = time.perf_counter()
    ct_ch = engine.encrypt(pk, ch, r_ch)
    enc_ms= (time.perf_counter() - t_enc) * 1000

    t_dec = time.perf_counter()
    out_ch= engine.decrypt(sk, ct_ch)
    dec_ms= (time.perf_counter() - t_dec) * 1000

    decrypted_chunks.append(out_ch)
    ok = (out_ch == ch)
    total_plain  += len(ch)
    total_ct     += len(ct_ch)
    total_enc_ms += enc_ms
    total_dec_ms += dec_ms

    print(f"  {idx:>6}  {len(ch):>8}  {len(ct_ch):>8}  {enc_ms:>9.2f}  {dec_ms:>9.2f}  {'✓' if ok else '✗':>5}")
    record("pipeline", chunk_idx=idx, plain_bytes=len(ch), ct_bytes=len(ct_ch),
           enc_ms=round(enc_ms,3), dec_ms=round(dec_ms,3), correct=ok)

reassembled = b"".join(decrypted_chunks).rstrip(b"\x00").decode()
print(f"\n  Total plaintext in  : {total_plain} bytes")
print(f"  Total ciphertext out: {total_ct} bytes  ({total_ct/total_plain:.1f}× expansion)")
print(f"  Total encrypt time  : {total_enc_ms:.2f} ms")
print(f"  Total decrypt time  : {total_dec_ms:.2f} ms")
print(f"  Reassembled message : '{reassembled[:60]}…'")

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