# =============================================================================
# scmarv32 — Full 64-word test program
#
# What this computes:
#   1. fibonacci(8) stored in x3  (should be 21 = 0x15)
#   2. popcount(x3) stored in x4  (count set bits of 21 = 3)
#   3. factorial(5) stored in x5  (should be 120 = 0x78)
#   4. stores results to data memory and loads them back
#   5. self-halts
#
# Register usage:
#   x1  = scratch / loop counter
#   x2  = scratch / loop counter
#   x3  = fibonacci result
#   x4  = popcount result
#   x5  = factorial result
#   x6  = temp
#   x7  = temp
#   x8  = mem base address (always 0)
# =============================================================================

# ── Section 1: Fibonacci(8) ──────────────────────────────────────────────────
# fib(0)=0 fib(1)=1 fib(2)=1 ... fib(8)=21
# x3 = fib(n),  x6 = fib(n-1),  x1 = loop counter

        addi    x3, x0, 0          # x3 = fib(0) = 0
        addi    x6, x0, 1          # x6 = fib(1) = 1
        addi    x1, x0, 7          # x1 = 7 iterations (fib 1->8)

fib_loop:
        add     x7, x3, x6         # x7  = fib(n-1) + fib(n-2)
        addi    x3, x6, 0          # x3  = old x6  (mv x3, x6)
        addi    x6, x7, 0          # x6  = new fib (mv x6, x7)
        addi    x1, x1, -1         # x1-- 
        bne     x1, x0, fib_loop   # loop if x1 != 0

# x6 = fib(8) = 21, move into x3
        addi    x3, x6, 0          # x3 = fib(8) = 21

# ── Section 2: Popcount(x3) ─────────────────────────────────────────────────
# Count number of set bits in x3 (21 = 0b10101 -> 3 bits set)
# Algorithm: while x2 != 0 { if (x2 & 1) x4++ ; x2 >>= 1 }

        addi    x4, x0, 0          # x4 = 0  (result)
        addi    x2, x3, 0          # x2 = copy of x3 to shift down
        addi    x6, x0, 1          # x6 = 1  (mask)

pop_loop:
        and     x7, x2, x6         # x7 = x2 & 1
        add     x4, x4, x7         # x4 += lsb
        srli    x2, x2, 1          # x2 >>= 1
        bne     x2, x0, pop_loop   # loop while bits remain

# ── Section 3: Factorial(5) ─────────────────────────────────────────────────
# x5 = 5! = 120
# x5 = accumulator, x1 = counter 5 down to 1

        addi    x5, x0, 1          # x5 = 1
        addi    x1, x0, 5          # x1 = 5

# Multiply x5 * x1 using repeated addition into x5
# Since we have no MUL, implement as:
#   for i = n downto 1: x5 = x5 * i  using add-loop

fact_outer:
        addi    x6, x0, 0          # x6 = 0  (product accumulator)
        addi    x2, x1, 0          # x2 = x1 (repeat x1 times)

fact_inner:
        add     x6, x6, x5         # x6 += x5
        addi    x2, x2, -1         # x2--
        bne     x2, x0, fact_inner # loop x1 times

        addi    x5, x6, 0          # x5 = x6 (new product)
        addi    x1, x1, -1         # x1--
        bne     x1, x0, fact_outer # next factor

# ── Section 4: Bitwise ops on results ───────────────────────────────────────
# Just to exercise more instructions

        and     x6, x3, x4         # x6 = fib & popcount  (21 & 3 = 1)
        or      x6, x6, x5         # x6 = x6 | factorial  (1 | 120 = 121)
        xor     x7, x3, x5         # x7 = fib ^ factorial (21 ^ 120 = 109)
        slt     x1, x3, x5         # x1 = (21 < 120) = 1
        sltu    x2, x4, x3         # x2 = (3 <u 21)  = 1

# ── Section 5: Shifts ────────────────────────────────────────────────────────

        slli    x6, x3, 2          # x6 = 21 << 2 = 84
        srli    x7, x5, 1          # x7 = 120 >> 1 = 60
        srai    x1, x5, 2          # x1 = 120 >>> 2 = 30  (arithmetic)
        sll     x2, x3, x4         # x2 = 21 << 3 = 168
        srl     x6, x5, x4         # x6 = 120 >> 3 = 15
        sra     x7, x5, x4         # x7 = 120 >>> 3 = 15

# ── Section 6: Memory store and load ────────────────────────────────────────
# Store fib, popcount, factorial to addresses 0, 4, 8

        addi    x8, x0, 0          # base address = 0
        sw      x3, 0(x8)          # mem[0] = fib(8)    = 21
        sw      x4, 4(x8)          # mem[4] = popcount  = 3
        sw      x5, 8(x8)          # mem[8] = factorial = 120

        lw      x1, 0(x8)          # x1 = mem[0] = 21
        lw      x2, 4(x8)          # x2 = mem[4] = 3
        lw      x6, 8(x8)          # x6 = mem[8] = 120

# Verify load by re-adding: x7 = x1 + x2 + x6 = 21 + 3 + 120 = 144
        add     x7, x1, x2         # x7 = 24
        add     x7, x7, x6         # x7 = 144 = 0x90

# ── Section 7: LUI and AUIPC ─────────────────────────────────────────────────
#
#         lui     x1, 1              # x1 = 0x00001000
#         auipc   x2, 0              # x2 = current PC (just to exercise it)
#         add     x6, x1, x3         # x6 = 0x1000 + 21 = 0x1015

# ── Section 8: BEQ test ──────────────────────────────────────────────────────

        addi    x1, x0, 42         # x1 = 42
        addi    x2, x0, 42         # x2 = 42
        beq     x1, x2, eq_taken   # should be taken
        addi    x7, x0, 0          # skipped if branch taken

eq_taken:
        addi    x7, x0, 99         # x7 = 99 (beq worked)

# ── Section 9: JALR ──────────────────────────────────────────────────────────

        jal     x1, jalr_target    # jump, x1 = return addr
        addi    x6, x0, 0          # skipped
        addi    x6, x0, 0          # skipped

jalr_target:
        addi    x6, x0, 55         # x6 = 55 (jal worked)
        jalr    x0, x1, 0          # return (jump back to after jal)

# ── Section 10: Final accumulation and halt ──────────────────────────────────
# Combine everything into x3 as final checksum, then halt

        add     x3, x3, x4         # x3 = 21 + 3 = 24
        add     x3, x3, x5         # x3 = 24 + 120 = 144 = 0x90
        xor     x3, x3, x7         # mix in x7
        addi    x3, x3, 1          # +1

halt:
        jal     x0, halt           # infinite loop — halt
