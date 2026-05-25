if __name__ == "__main__":
    import os
    from .KyberCore import KyberCoreEngine
    import hashlib
    
    # Global Config Settings
    N_PARAM = 256
    Q_PARAM = 3329
    ETA_PARAM = 2
    
    print("=" * 65)
    print(f" Simulating Modular FPGA Architecture: N={N_PARAM}, Q={Q_PARAM}")
    print("=" * 65)

    # Initialize Core Modules exactly as they would map to separate FPGA IP Blocks
    kyber_512_hardware = KyberCoreEngine(N_PARAM, Q_PARAM, ETA_PARAM, k_dimension=2)
    
    # 1. Simulate KeyGen Pipeline
    dev_seed = os.urandom(32)
    pk, sk = kyber_512_hardware.keygen(dev_seed)
    print(" [+] Key Generation Module Executed successfully.")

    # 2. Input Stream Simulation Block
    input_string = "Hello World! Processing modular code block layout."
    raw_bytes = input_string.encode('utf-8')
    
    # Slice the stream into chunk packages matching hardware data paths
    chunk_bytes_len = kyber_512_hardware.serializer.chunk_size
    chunks = [raw_bytes[i:i+chunk_bytes_len] for i in range(0, len(raw_bytes), chunk_bytes_len)]
    if chunks and len(chunks[-1]) < chunk_bytes_len:
        chunks[-1] = chunks[-1].ljust(chunk_bytes_len, b'\x00')

    # 3. Stream through Encryption and Decryption engines
    ciphertexts = []
    for idx, plain_chunk in enumerate(chunks):
        ephemeral_seed = hashlib.sha256(b"seed" + idx.to_bytes(4, "big")).digest()
        ct_packet = kyber_512_hardware.encrypt(pk, plain_chunk, ephemeral_seed)
        ciphertexts.append(ct_packet)
    print(f" [+] Encryption Pipeline processed {len(chunks)} text package chunks.")

    decrypted_chunks = []
    for ct_packet in ciphertexts:
        recovered_chunk = kyber_512_hardware.decrypt(sk, ct_packet)
        decrypted_chunks.append(recovered_chunk)
        
    reassembled_output = b"".join(decrypted_chunks).rstrip(b"\x00").decode('utf-8')
    print(f" [+] Decryption Reassembly Completed.")
    print(f"     Output: '{reassembled_output}'\n")
