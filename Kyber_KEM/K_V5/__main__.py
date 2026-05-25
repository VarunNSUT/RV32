if __name__ == "__main__":
    import os
    import time
    import hashlib
    from .KyberCore import KyberCoreEngine

    # ── Global Config ────────────────────────────────────────────────────────
    N_PARAM   = 256
    Q_PARAM   = 3329
    ETA_PARAM = 2

    # Kyber variant → k dimension
    VARIANTS = {512: 2, 768: 3, 1024: 4}

    # Compression params per variant  (du, dv)
    COMPRESS_PARAMS = {
        512:  (10, 4),
        768:  (10, 4),
        1024: (11, 5),
    }

    # ── Test messages ────────────────────────────────────────────────────────
    TEST_MESSAGES = {
        "short": "Hi!",
        "exact": "A" * 32,       # exactly one chunk for N=256
        "long": (
            "Hello World! Standard cryptographic text verification data. "
            "This message is intentionally long to exercise the chunking path. "
            "Kyber encrypts it block by block and recovers every character."
        ),
    }

    print("=" * 65)
    print(f"  Modular FPGA Kyber Architecture: N={N_PARAM}, Q={Q_PARAM}")
    print("=" * 65)

    for variant, k_dim in VARIANTS.items():
        du, dv = COMPRESS_PARAMS[variant]
        print(f"\n── Kyber-{variant} (k={k_dim}, du={du}, dv={dv}) " + "─" * 20)

        engine = KyberCoreEngine(N_PARAM, Q_PARAM, ETA_PARAM, k_dimension=k_dim)
        engine._du = du
        engine._dv = dv

        # KeyGen
        t0 = time.perf_counter()
        pk, sk = engine.keygen(os.urandom(32))
        t_keygen = time.perf_counter() - t0

        chunk_size = engine.serializer.chunk_size

        for label, msg_str in TEST_MESSAGES.items():
            raw_bytes = msg_str.encode("utf-8")

            # Chunk + pad
            chunks = [
                raw_bytes[i : i + chunk_size]
                for i in range(0, max(len(raw_bytes), 1), chunk_size)
            ]
            if len(chunks[-1]) < chunk_size:
                chunks[-1] = chunks[-1].ljust(chunk_size, b"\x00")

            # Encrypt
            t0 = time.perf_counter()
            ciphertexts = []
            for idx, chunk in enumerate(chunks):
                r_seed = hashlib.sha256(b"seed" + idx.to_bytes(4, "big")).digest()
                ciphertexts.append(engine.encrypt(pk, chunk, r_seed))
            t_enc = time.perf_counter() - t0

            # Decrypt
            t0 = time.perf_counter()
            recovered = b"".join(
                engine.decrypt(sk, ct) for ct in ciphertexts
            ).rstrip(b"\x00").decode("utf-8")
            t_dec = time.perf_counter() - t0

            ok = "✓ PASS" if recovered == msg_str else "✗ FAIL"
            print(f"\n  [{label}]  chunks={len(chunks)}  {ok}")
            print(f"  Original : {repr(msg_str[:60])}{'...' if len(msg_str) > 60 else ''}")
            print(f"  Recovered: {repr(recovered[:60])}{'...' if len(recovered) > 60 else ''}")
            print(f"  KeyGen={t_keygen*1000:.1f}ms  Encrypt={t_enc*1000:.1f}ms  Decrypt={t_dec*1000:.1f}ms")

    # ── Noise Analysis ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Noise Evaluation Module  (Kyber-512)")
    print("=" * 65)

    from kyber.ArithmeticUnit import ntt, intt, ntt_mul
    engine = KyberCoreEngine(N_PARAM, Q_PARAM, ETA_PARAM, k_dimension=2)
    engine._du, engine._dv = 10, 4
    pk, sk = engine.keygen(os.urandom(32))

    msg    = os.urandom(engine.serializer.chunk_size)
    r_seed = os.urandom(32)

    # Encrypt then inspect the raw decryption polynomial before decode
    rho, t = pk
    AT_ntt = engine.vmm.mat_transpose(engine._get_A(rho))
    r      = [engine.sampler.sample_cbd(r_seed, i)           for i in range(engine.k)]
    e1     = [engine.sampler.sample_cbd(r_seed, engine.k+i)  for i in range(engine.k)]
    e2     = engine.sampler.sample_cbd(r_seed, 2 * engine.k)
    r_ntt  = [ntt(ri) for ri in r]
    t_ntt  = [ntt(ti) for ti in t]

    u  = engine.vmm.vec_add(
             engine.vmm.mat_vec_mul(AT_ntt, r_ntt, A_in_ntt=True, v_in_ntt=True), e1)
    tr = engine.vmm.vec_dot(t_ntt, r_ntt, a_in_ntt=True, b_in_ntt=True)
    v  = engine.alu.poly_add(
             engine.alu.poly_add(tr, e2),
             engine.serializer.encode_message_to_poly(msg))

    s_ntt = [ntt(si) for si in sk]
    u_ntt = [ntt(ui) for ui in u]
    w     = engine.alu.poly_sub(v, engine.vmm.vec_dot(s_ntt, u_ntt, a_in_ntt=True, b_in_ntt=True))
    m_bar = engine.serializer.encode_message_to_poly(msg)

    noise          = [(int(w[i]) - int(m_bar[i])) % Q_PARAM for i in range(N_PARAM)]
    noise_centered = [engine.alu.centered_mod(n) for n in noise]
    max_noise      = max(abs(n) for n in noise_centered)

    print(f"  q/4 error correction ceiling:  {Q_PARAM // 4}")
    print(f"  Max Absolute Observed Noise:   {max_noise}  "
          f"({'within bound' if max_noise < Q_PARAM // 4 else 'EXCEEDED — decryption failure'})")
    print(f"  Noise Distribution (first 10): {[int(n) for n in noise_centered[:10]]}")
    print()