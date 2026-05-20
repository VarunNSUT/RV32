`timescale 1ns / 1ps

module tb_riscv_debug;

  reg clk;
  reg rst;

  integer fd; // file descriptor

  scmarv32_top uut (
    .clk(clk),
    .rst(rst)
  );

  always #5 clk = ~clk;

  // ──────────────────────────────────────────────
  // Helper function: decode opcode to string
  // ──────────────────────────────────────────────
  function [79:0] decode_opcode; // 10 chars * 8 bits
    input [6:0] opcode;
    input [2:0] funct3;
    input [6:0] funct7;
    begin
      case (opcode)
        7'b0110011: begin
          case (funct3)
            3'b000: decode_opcode = funct7[5] ? "SUB       " : "ADD       ";
            3'b111: decode_opcode = "AND       ";
            3'b110: decode_opcode = "OR        ";
            3'b100: decode_opcode = "XOR       ";
            3'b001: decode_opcode = "SLL       ";
            3'b101: decode_opcode = funct7[5] ? "SRA       " : "SRL       ";
            3'b010: decode_opcode = "SLT       ";
            3'b011: decode_opcode = "SLTU      ";
            default: decode_opcode = "R-UNKN    ";
          endcase
        end
        7'b0010011: begin
          case (funct3)
            3'b000: decode_opcode = "ADDI      ";
            3'b111: decode_opcode = "ANDI      ";
            3'b110: decode_opcode = "ORI       ";
            3'b100: decode_opcode = "XORI      ";
            3'b001: decode_opcode = "SLLI      ";
            3'b101: decode_opcode = funct7[5] ? "SRAI      " : "SRLI      ";
            3'b010: decode_opcode = "SLTI      ";
            3'b011: decode_opcode = "SLTIU     ";
            default: decode_opcode = "I-UNKN    ";
          endcase
        end
        7'b0000011: decode_opcode = "LW        ";
        7'b0100011: decode_opcode = "SW        ";
        7'b1100011: begin
          case (funct3)
            3'b000: decode_opcode = "BEQ       ";
            3'b001: decode_opcode = "BNE       ";
            3'b100: decode_opcode = "BLT       ";
            3'b101: decode_opcode = "BGE       ";
            default: decode_opcode = "BR-UNKN   ";
          endcase
        end
        7'b0110111: decode_opcode = "LUI       ";
        7'b0010111: decode_opcode = "AUIPC     ";
        7'b1101111: decode_opcode = "JAL       ";
        7'b1100111: decode_opcode = "JALR      ";
        default:    decode_opcode = "UNKNOWN   ";
      endcase
    end
  endfunction

  // ──────────────────────────────────────────────
  // Main simulation
  // ──────────────────────────────────────────────
  initial begin
    $dumpfile("simulation.vcd");
    $dumpvars(0, tb_riscv_debug);

    fd = $fopen("debug_log.txt", "w");
    if (fd == 0) begin
      $display("ERROR: Could not open debug_log.txt");
      $finish;
    end

    // ── Header ──
    $fwrite(fd, "=======================================================================================================================================\n");
    $fwrite(fd, " RISC-V Single Cycle Debug Log\n");
    $fwrite(fd, "=======================================================================================================================================\n");
    $fwrite(fd, "\n");

    // ── Control signals header ──
    $fwrite(fd, "--- CONTROL SIGNALS KEY ---\n");
    $fwrite(fd, "  RW=reg_write  IS=imm_src  AS=alu_src  MW=mem_write  MR=mem_read  M2R=mem_to_reg  BR=branch\n");
    $fwrite(fd, "  ALU_CTL: 0000=AND 0001=OR 0010=ADD 0011=XOR 0100=SLL 0101=SRL 0110=SUB 0111=SRA 1000=SLT 1001=SLTU\n");
    $fwrite(fd, "\n");

    // ── Column headers ──
    $fwrite(fd, "%-10s | %-8s | %-8s | %-10s | %-8s | %-8s | RW IS AS MW MR M2R BR | %-7s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s | %-8s\n",
      "Time(ns)", "PC", "Instr", "Mnemonic", "rs1(val)", "rs2(val)",
      "ALU_CTL", "IMM_EXT", "ALU_OUT", "ZERO",
      "x1", "x2", "x3", "x4", "x5");
    $fwrite(fd, "%s\n", {200{"-"}});

    clk = 0;
    rst = 1;
    #12;
    rst = 0;
    #5000

    // ── Footer ──
    $fwrite(fd, "\n");
    $fwrite(fd, "=======================================================================================================================================\n");
    $fwrite(fd, " Register File Final State\n");
    $fwrite(fd, "=======================================================================================================================================\n");
    $fwrite(fd, "  x0  (zero) = %08h\n", uut.rf.rf[0]);
    $fwrite(fd, "  x1  (ra)   = %08h\n", uut.rf.rf[1]);
    $fwrite(fd, "  x2  (sp)   = %08h\n", uut.rf.rf[2]);
    $fwrite(fd, "  x3  (gp)   = %08h\n", uut.rf.rf[3]);
    $fwrite(fd, "  x4  (tp)   = %08h\n", uut.rf.rf[4]);
    $fwrite(fd, "  x5  (t0)   = %08h\n", uut.rf.rf[5]);
    $fwrite(fd, "  x6  (t1)   = %08h\n", uut.rf.rf[6]);
    $fwrite(fd, "  x7  (t2)   = %08h\n", uut.rf.rf[7]);
    $fwrite(fd, "  x8  (s0)   = %08h\n", uut.rf.rf[8]);
    $fwrite(fd, "  x9  (s1)   = %08h\n", uut.rf.rf[9]);
    $fwrite(fd, "  x10 (a0)   = %08h\n", uut.rf.rf[10]);
    $fwrite(fd, "  x11 (a1)   = %08h\n", uut.rf.rf[11]);
    $fwrite(fd, "=======================================================================================================================================\n");
    $fwrite(fd, " Final PC = %08h\n", uut.pc_current);
    $fwrite(fd, "=======================================================================================================================================\n");

    $fclose(fd);
    $display("Debug log written to debug_log.txt");
    $finish;
  end

  // ──────────────────────────────────────────────
  // Per-cycle logging — fires on every rising edge
  // after reset deasserts
  // ──────────────────────────────────────────────
  always @(posedge clk) begin
    if (!rst) begin
      // Small delay so registered values have settled
      #1;
      $fwrite(fd, "%-10t | %08h | %08h | %-10s | x%0d=%05h | x%0d=%05h |  %b  %b  %b  %b  %b   %b   %b  | %04b    | %08h | %08h | %b    | %08h | %08h | %08h | %08h | %08h\n",
        $time,
        uut.pc_current,
        uut.instr,
        decode_opcode(uut.instr[6:0], uut.instr[14:12], uut.instr[31:25]),
        uut.instr[19:15], uut.rd1,          // rs1 index + value
        uut.instr[24:20], uut.rd2,          // rs2 index + value
        // Control signals
        uut.reg_write,
        uut.imm_src,
        uut.alu_src,
        uut.mem_write,
        uut.mem_read,
        uut.mem_to_reg,
        uut.branch,
        uut.alu_control,
        uut.imm_ext,
        uut.alu_result,
        uut.alu_zero,
        // Key registers
        uut.rf.rf[1],
        uut.rf.rf[2],
        uut.rf.rf[3],
        uut.rf.rf[4],
        uut.rf.rf[5]
      );
    end
  end

endmodule
