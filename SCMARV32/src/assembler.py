#!/usr/bin/env python3
"""
scmarv32 Assembler
Converts RISC-V32 assembly to hex for the scmarv32 single-cycle processor.

Supported instructions (matching your control_unit):
  R-type : add, sub, and, or, xor, sll, srl, sra, slt, sltu
  I-type : addi, andi, ori, xori, slli, srli, srai, slti, sltiu
  Load   : lw
  Store  : sw
  Branch : beq, bne
  Jump   : jal
  Upper  : lui, auipc

Usage:
  python3 assembler.py program.asm              -> writes program.hex
  python3 assembler.py program.asm -o out.hex   -> custom output file
  python3 assembler.py program.asm --verbose    -> show decoded table
"""

import re
import argparse


# ─── Register map ────────────────────────────────────────────────────────────

REGISTERS = {
    # ABI names
    "zero": 0,  "ra": 1,   "sp": 2,   "gp": 3,
    "tp":   4,  "t0": 5,   "t1": 6,   "t2": 7,
    "s0":   8,  "fp": 8,   "s1": 9,   "a0": 10,
    "a1":   11, "a2": 12,  "a3": 13,  "a4": 14,
    "a5":   15, "a6": 16,  "a7": 17,  "s2": 18,
    "s3":   19, "s4": 20,  "s5": 21,  "s6": 22,
    "s7":   23, "s8": 24,  "s9": 25,  "s10": 26,
    "s11":  27, "t3": 28,  "t4": 29,  "t5": 30,
    "t6":   31,
    # x0-x31 numeric names
    **{f"x{i}": i for i in range(32)},
}

# ─── Instruction encodings ────────────────────────────────────────────────────

R_TYPE = {
    #  mnemonic : (funct3, funct7)
    "add":  (0b000, 0b0000000),
    "sub":  (0b000, 0b0100000),
    "and":  (0b111, 0b0000000),
    "or":   (0b110, 0b0000000),
    "xor":  (0b100, 0b0000000),
    "sll":  (0b001, 0b0000000),
    "srl":  (0b101, 0b0000000),
    "sra":  (0b101, 0b0100000),
    "slt":  (0b010, 0b0000000),
    "sltu": (0b011, 0b0000000),
}

I_ARITH = {
    #  mnemonic : funct3   (funct7 only needed for srli/srai)
    "addi":  0b000,
    "andi":  0b111,
    "ori":   0b110,
    "xori":  0b100,
    "slli":  0b001,
    "srli":  0b101,
    "srai":  0b101,
    "slti":  0b010,
    "sltiu": 0b011,
}

BRANCH = {
    "beq": 0b000,
    "bne": 0b001,
    "blt": 0b100,
    "bge": 0b101,
}

OPCODES = {
    "R":      0b0110011,
    "I_ARITH":0b0010011,
    "LOAD":   0b0000011,
    "STORE":  0b0100011,
    "BRANCH": 0b1100011,
    "JAL":    0b1101111,
    "JALR":   0b1100111,
    "LUI":    0b0110111,
    "AUIPC":  0b0010111,
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def sign_extend(value, bits):
    """Sign-extend a value to a given bit width."""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def to_signed(value, bits):
    """Ensure value fits in signed immediate."""
    limit = 1 << (bits - 1)
    if not (-limit <= value < limit):
        raise ValueError(f"Immediate {value} out of range for {bits}-bit signed field")
    return value & ((1 << bits) - 1)  # return as unsigned bits


def parse_reg(token, context=""):
    token = token.strip().rstrip(",").lower()
    if token not in REGISTERS:
        raise ValueError(f"Unknown register '{token}'" + (f" in '{context}'" if context else ""))
    return REGISTERS[token]


def parse_imm(token, labels, pc, context=""):
    """Parse an immediate: decimal, hex (0x...), or label."""
    token = token.strip().rstrip(",")
    if token in labels:
        return labels[token] - pc   # PC-relative for branches/jal
    try:
        return int(token, 0)        # handles 0x hex and decimal
    except ValueError:
        raise ValueError(f"Unknown immediate or label '{token}'" + (f" in '{context}'" if context else ""))


def parse_mem_operand(token):
    """Parse 'imm(rs1)' -> (imm_str, rs1_str)."""
    m = re.match(r'^(-?\w+)\((\w+)\)$', token.strip())
    if not m:
        raise ValueError(f"Expected 'imm(reg)', got '{token}'")
    return m.group(1), m.group(2)


# ─── Encoders ─────────────────────────────────────────────────────────────────

def encode_r(rd, rs1, rs2, funct3, funct7):
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | OPCODES["R"]


def encode_i(rd, rs1, imm, funct3, opcode):
    imm12 = to_signed(imm, 12)
    return (imm12 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def encode_s(rs1, rs2, imm, funct3):
    imm12 = to_signed(imm, 12)
    imm_11_5 = (imm12 >> 5) & 0x7F
    imm_4_0  = imm12 & 0x1F
    return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (imm_4_0 << 7) | OPCODES["STORE"]


def encode_b(rs1, rs2, imm, funct3):
    if imm % 2 != 0:
        raise ValueError(f"Branch offset {imm} must be a multiple of 2")
    imm13 = to_signed(imm, 13)
    b12    = (imm13 >> 12) & 1
    b11    = (imm13 >> 11) & 1
    b10_5  = (imm13 >> 5)  & 0x3F
    b4_1   = (imm13 >> 1)  & 0xF
    return (b12 << 31) | (b10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (b4_1 << 8) | (b11 << 7) | OPCODES["BRANCH"]


def encode_j(rd, imm):
    if imm % 2 != 0:
        raise ValueError(f"JAL offset {imm} must be a multiple of 2")
    imm21  = to_signed(imm, 21)
    b20    = (imm21 >> 20) & 1
    b10_1  = (imm21 >> 1)  & 0x3FF
    b11    = (imm21 >> 11) & 1
    b19_12 = (imm21 >> 12) & 0xFF
    return (b20 << 31) | (b19_12 << 12) | (b11 << 20) | (b10_1 << 21) | (rd << 7) | OPCODES["JAL"]


def encode_u(rd, imm, opcode):
    imm20 = (imm >> 12) & 0xFFFFF
    return (imm20 << 12) | (rd << 7) | opcode


# ─── Pass 1: collect labels ───────────────────────────────────────────────────

def first_pass(lines):
    """Return (clean_lines, labels) where clean_lines has comments/blanks removed."""
    labels  = {}
    clean   = []
    pc      = 0

    for lineno, raw in enumerate(lines, 1):
        line = raw.split("#")[0].strip()   # strip comments
        if not line:
            continue

        if line.endswith(":"):             # label definition
            name = line[:-1].strip()
            if name in labels:
                raise SyntaxError(f"Line {lineno}: duplicate label '{name}'")
            labels[name] = pc
            continue

        # label on same line as instruction  e.g.  "loop: add x1, x2, x3"
        if ":" in line:
            parts = line.split(":", 1)
            name  = parts[0].strip()
            rest  = parts[1].strip()
            labels[name] = pc
            if rest:
                clean.append((lineno, pc, rest))
                pc += 4
            continue

        clean.append((lineno, pc, line))
        pc += 4

    return clean, labels


# ─── Pass 2: assemble ─────────────────────────────────────────────────────────

def assemble_line(lineno, pc, line, labels):
    tokens = re.split(r'[\s,]+', line.strip())
    tokens = [t for t in tokens if t]
    mnem   = tokens[0].lower()

    try:
        # ── R-type ──────────────────────────────────────────────────────────
        if mnem in R_TYPE:
            if len(tokens) != 4:
                raise ValueError(f"Expected: {mnem} rd, rs1, rs2")
            rd, rs1, rs2 = parse_reg(tokens[1]), parse_reg(tokens[2]), parse_reg(tokens[3])
            f3, f7 = R_TYPE[mnem]
            return encode_r(rd, rs1, rs2, f3, f7)

        # ── I-type arithmetic ────────────────────────────────────────────────
        elif mnem in I_ARITH:
            if len(tokens) != 4:
                raise ValueError(f"Expected: {mnem} rd, rs1, imm")
            rd  = parse_reg(tokens[1])
            rs1 = parse_reg(tokens[2])
            imm = parse_imm(tokens[3], labels, pc)
            f3  = I_ARITH[mnem]
            f7  = 0b0100000 if mnem == "srai" else 0b0000000
            # for shifts the imm is only 5 bits (shamt), pack funct7 into upper bits
            if mnem in ("slli", "srli", "srai"):
                shamt = imm & 0x1F
                imm12 = (f7 << 5) | shamt
                return (imm12 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | OPCODES["I_ARITH"]
            return encode_i(rd, rs1, imm, f3, OPCODES["I_ARITH"])

        # ── Load word ────────────────────────────────────────────────────────
        elif mnem == "lw":
            if len(tokens) != 3:
                raise ValueError("Expected: lw rd, imm(rs1)")
            rd       = parse_reg(tokens[1])
            imm_s, rs1_s = parse_mem_operand(tokens[2])
            rs1      = parse_reg(rs1_s)
            imm      = int(imm_s, 0)
            return encode_i(rd, rs1, imm, 0b010, OPCODES["LOAD"])

        # ── Store word ───────────────────────────────────────────────────────
        elif mnem == "sw":
            if len(tokens) != 3:
                raise ValueError("Expected: sw rs2, imm(rs1)")
            rs2      = parse_reg(tokens[1])
            imm_s, rs1_s = parse_mem_operand(tokens[2])
            rs1      = parse_reg(rs1_s)
            imm      = int(imm_s, 0)
            return encode_s(rs1, rs2, imm, 0b010)

        # ── Branch ───────────────────────────────────────────────────────────
        elif mnem in BRANCH:
            if len(tokens) != 4:
                raise ValueError(f"Expected: {mnem} rs1, rs2, label/imm")
            rs1 = parse_reg(tokens[1])
            rs2 = parse_reg(tokens[2])
            imm = parse_imm(tokens[3], labels, pc)
            return encode_b(rs1, rs2, imm, BRANCH[mnem])

        # ── JAL ──────────────────────────────────────────────────────────────
        elif mnem == "jal":
            if len(tokens) != 3:
                raise ValueError("Expected: jal rd, label/imm")
            rd  = parse_reg(tokens[1])
            imm = parse_imm(tokens[2], labels, pc)
            return encode_j(rd, imm)

        # ── JALR ─────────────────────────────────────────────────────────────
        elif mnem == "jalr":
            if len(tokens) == 4:
                # jalr rd, rs1, imm
                rd  = parse_reg(tokens[1])
                rs1 = parse_reg(tokens[2])
                imm = parse_imm(tokens[3], labels, pc)
            elif len(tokens) == 3:
                # jalr rd, imm(rs1)
                rd = parse_reg(tokens[1])
                imm_s, rs1_s = parse_mem_operand(tokens[2])
                rs1 = parse_reg(rs1_s)
                imm = int(imm_s, 0)
            else:
                raise ValueError("Expected: jalr rd, rs1, imm  OR  jalr rd, imm(rs1)")
            return encode_i(rd, rs1, imm, 0b000, OPCODES["JALR"])

        # ── LUI ──────────────────────────────────────────────────────────────
        elif mnem == "lui":
            if len(tokens) != 3:
                raise ValueError("Expected: lui rd, imm")
            rd  = parse_reg(tokens[1])
            imm = parse_imm(tokens[2], labels, pc)
            return encode_u(rd, imm, OPCODES["LUI"])

        # ── AUIPC ────────────────────────────────────────────────────────────
        elif mnem == "auipc":
            if len(tokens) != 3:
                raise ValueError("Expected: auipc rd, imm")
            rd  = parse_reg(tokens[1])
            imm = parse_imm(tokens[2], labels, pc)
            return encode_u(rd, imm, OPCODES["AUIPC"])

        # ── Pseudoinstructions ───────────────────────────────────────────────
        elif mnem == "nop":
            return encode_i(0, 0, 0, 0b000, OPCODES["I_ARITH"])   # addi x0, x0, 0

        elif mnem == "li":
            # li rd, imm  →  addi rd, x0, imm   (works for small imms)
            if len(tokens) != 3:
                raise ValueError("Expected: li rd, imm")
            rd  = parse_reg(tokens[1])
            imm = parse_imm(tokens[2], labels, pc)
            return encode_i(rd, 0, imm, 0b000, OPCODES["I_ARITH"])

        elif mnem == "mv":
            # mv rd, rs  →  addi rd, rs, 0
            if len(tokens) != 3:
                raise ValueError("Expected: mv rd, rs")
            rd  = parse_reg(tokens[1])
            rs1 = parse_reg(tokens[2])
            return encode_i(rd, rs1, 0, 0b000, OPCODES["I_ARITH"])

        elif mnem == "j":
            # j label  →  jal x0, label
            if len(tokens) != 2:
                raise ValueError("Expected: j label")
            imm = parse_imm(tokens[1], labels, pc)
            return encode_j(0, imm)

        elif mnem == "ret":
            # ret  →  jalr x0, x1, 0
            return encode_i(0, 1, 0, 0b000, OPCODES["JALR"])

        elif mnem == "call":
            # call label  →  jal ra, label
            if len(tokens) != 2:
                raise ValueError("Expected: call label")
            imm = parse_imm(tokens[1], labels, pc)
            return encode_j(1, imm)

        else:
            raise ValueError(f"Unknown mnemonic '{mnem}'")

    except ValueError as e:
        raise ValueError(f"Line {lineno} (PC=0x{pc:08x}): {e}") from None


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="scmarv32 RISC-V assembler")
    parser.add_argument("input",  help="Input assembly file (.asm)")
    parser.add_argument("-o", "--output", help="Output hex file (default: program.hex)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print decoded instruction table")
    args = parser.parse_args()

    out_file = args.output or "program.hex"

    with open(args.input) as f:
        lines = f.readlines()

    clean, labels = first_pass(lines)

    if args.verbose:
        print(f"{'Label':<12} {'PC':>10}  (from first pass)")
        for name, addr in labels.items():
            print(f"  {name:<10} 0x{addr:08x}")
        print()
        print(f"{'PC':<12} {'Hex':>10}  {'Binary':>34}  Instruction")
        print("-" * 80)

    instructions = []
    for lineno, pc, line in clean:
        word = assemble_line(lineno, pc, line, labels)
        instructions.append((pc, word, line.strip()))
        if args.verbose:
            print(f"0x{pc:08x}   {word:08x}   {word:032b}  {line.strip()}")

    with open(out_file, "w") as f:
        for _, word, _ in instructions:
            f.write(f"{word:08x}\n")

    print(f"Assembled {len(instructions)} instruction(s) -> {out_file}")


if __name__ == "__main__":
    main()
