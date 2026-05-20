module pc (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] pc_next,
    output reg  [31:0] pc
);
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      pc <= 32'h0000_0000;
    end else begin
      pc <= pc_next;
    end
  end
endmodule


module pc_adder (
    input  wire [31:0] pc,
    output wire [31:0] pc_next
);
  assign pc_next = pc + 32'd4;
endmodule


module instruction_reg (
    input  wire [31:0] A,
    output wire [31:0] instruction
);
  reg [31:0] rom[256];

  initial begin
    $readmemh("program.hex", rom);
  end

  assign instruction = rom[A[9:2]];
endmodule


module reg_file (
    input  wire        clk,
    input  wire        rst,
    input  wire        reg_write,
    input  wire [ 4:0] rs1,
    input  wire [ 4:0] rs2,
    input  wire [ 4:0] rd,
    input  wire [31:0] wd,
    output wire [31:0] rd1,
    output wire [31:0] rd2
);
  reg [31:0] rf[0:31];
  integer i;

  always @(negedge clk) begin  // ← negedge instead of posedge
    if (rst) begin
      for (i = 0; i < 32; i = i + 1) rf[i] <= 32'h0000_0000;
    end else if (reg_write && (rd != 5'b00000)) begin
      rf[rd] <= wd;
    end
  end

  // Reads stay simple — no forwarding needed
  assign rd1 = (rs1 == 5'b00000) ? 32'h0 : rf[rs1];
  assign rd2 = (rs2 == 5'b00000) ? 32'h0 : rf[rs2];
endmodule


module imm_gen (
    input  wire [31:0] instruction,
    output reg  [31:0] imm_ext
);
  wire [6:0] opcode = instruction[6:0];

  always @(*) begin
    case (opcode)
      7'b0010011, 7'b0000011: begin
        imm_ext = {{20{instruction[31]}}, instruction[31:20]};
      end

      7'b0100011: begin
        imm_ext = {{20{instruction[31]}}, instruction[31:25], instruction[11:7]};
      end

      7'b1100011: begin
        imm_ext = {
          {20{instruction[31]}}, instruction[7], instruction[30:25], instruction[11:8], 1'b0
        };
      end

      7'b0110111, 7'b0010111: begin
        imm_ext = {instruction[31:12], 12'h000};
      end

      7'b1101111: begin
        imm_ext = {
          {12{instruction[31]}}, instruction[19:12], instruction[20], instruction[30:21], 1'b0
        };
      end

      default: begin
        imm_ext = 32'h0000_0000;
      end
    endcase
  end
endmodule


module alu (
    input  wire [31:0] A,
    input  wire [31:0] B,
    input  wire [ 3:0] alu_control,
    output reg  [31:0] alu_output,
    output wire        zero
);
  always @(*) begin
    case (alu_control)
      4'b0000: alu_output = A & B;
      4'b0001: alu_output = A | B;
      4'b0010: alu_output = A + B;
      4'b0110: alu_output = A - B;
      4'b0011: alu_output = A ^ B;
      4'b0100: alu_output = A << B[4:0];
      4'b0101: alu_output = A >> B[4:0];
      4'b0111: alu_output = $signed(A) >>> B[4:0];
      4'b1000: alu_output = ($signed(A) < $signed(B)) ? 32'd1 : 32'd0;
      4'b1001: alu_output = (A < B) ? 32'd1 : 32'd0;
      default: alu_output = 32'h0000_0000;
    endcase
  end

  assign zero = (alu_output == 32'h0000_0000) ? 1'b1 : 1'b0;
endmodule


module data_mem (
    input  wire        clk,
    input  wire        rst,
    input  wire        mem_read,
    input  wire        mem_write,
    input  wire [31:0] addr,
    input  wire [31:0] write_data,
    output wire [31:0] read_data
);
  reg [31:0] ram[64];
  integer i;

  always @(posedge clk) begin
    if (rst) begin
      for (i = 0; i < 64; i = i + 1) begin
        ram[i] <= 32'h0000_0000;
      end
    end else if (mem_write) begin
      ram[addr[7:2]] <= write_data;
    end
  end

  assign read_data = (mem_read) ? ram[addr[7:2]] : 32'h0000_0000;
endmodule


module control_unit (
    input  wire [6:0] opcode,
    input  wire [2:0] funct3,
    input  wire [6:0] funct7,
    output reg        reg_write,
    output reg        imm_src,
    output reg        alu_src,
    output reg        mem_write,
    output reg        mem_read,
    output reg        mem_to_reg,
    output reg        branch,
    output reg        jal,
    output reg  [3:0] alu_control
);

  always @(*) begin
    // Explicit default states initialized every single execution pass
    jal         = 1'b0;
    reg_write   = 1'b0;
    imm_src     = 1'b0;
    alu_src     = 1'b0;
    mem_write   = 1'b0;
    mem_read    = 1'b0;
    mem_to_reg  = 1'b0;
    branch      = 1'b0;
    alu_control = 4'b0000;

    case (opcode)
      // R-type instructions (add, sub, and, or, xor, etc.)
      7'b0110011: begin
        reg_write  = 1'b1;
        imm_src    = 1'b0;
        alu_src    = 1'b0;
        mem_write  = 1'b0;
        mem_read   = 1'b0;
        mem_to_reg = 1'b0;
        branch     = 1'b0;
        case (funct3)
          3'b000:  alu_control = (funct7[5]) ? 4'b0110 : 4'b0010;  // sub : add
          3'b111:  alu_control = 4'b0000;  // and
          3'b110:  alu_control = 4'b0001;  // or
          3'b100:  alu_control = 4'b0011;  // xor
          3'b001:  alu_control = 4'b0100;  // sll
          3'b101:  alu_control = (funct7[5]) ? 4'b0111 : 4'b0101;  // sra : srl
          3'b010:  alu_control = 4'b1000;  // slt
          3'b011:  alu_control = 4'b1001;  // sltu
          default: alu_control = 4'b0000;
        endcase
      end

      // I-type Arithmetic (addi, andi, ori, etc.)
      7'b0010011: begin
        reg_write  = 1'b1;
        imm_src    = 1'b0;
        alu_src    = 1'b1;
        mem_write  = 1'b0;
        mem_read   = 1'b0;
        mem_to_reg = 1'b0;
        branch     = 1'b0;
        case (funct3)
          3'b000:  alu_control = 4'b0010;  // addi explicitly overrides upper flags
          3'b111:  alu_control = 4'b0000;  // andi
          3'b110:  alu_control = 4'b0001;  // ori
          3'b100:  alu_control = 4'b0011;  // xori
          3'b001:  alu_control = 4'b0100;  // slli
          3'b101:  alu_control = (funct7[5]) ? 4'b0111 : 4'b0101;  // srai : srli
          3'b010:  alu_control = 4'b1000;  // slti
          3'b011:  alu_control = 4'b1001;  // sltiu
          default: alu_control = 4'b0000;
        endcase
      end

      // Load Word (lw)
      7'b0000011: begin
        reg_write   = 1'b1;
        imm_src     = 1'b0;
        alu_src     = 1'b1;
        mem_write   = 1'b0;
        mem_read    = 1'b1;
        mem_to_reg  = 1'b1;
        branch      = 1'b0;
        alu_control = 4'b0010;  // Use ADD for address offset
      end

      // Store Word (sw)
      7'b0100011: begin
        reg_write   = 1'b0;
        imm_src     = 1'b0;
        alu_src     = 1'b1;
        mem_write   = 1'b1;
        mem_read    = 1'b0;
        mem_to_reg  = 1'b0;
        branch      = 1'b0;
        alu_control = 4'b0010;  // Use ADD for address offset
      end

      // Control Branches (beq, bne)
      7'b1100011: begin
        reg_write   = 1'b0;
        imm_src     = 1'b0;
        alu_src     = 1'b0;
        mem_write   = 1'b0;
        mem_read    = 1'b0;
        mem_to_reg  = 1'b0;
        branch      = 1'b1;
        alu_control = 4'b0110;  // Force SUBTRACT for comparison matches
      end

      //JAL case 
      7'b1101111: begin
        jal         = 1'b1;
        reg_write   = 1'b1;
        imm_src     = 1'b0;
        alu_src     = 1'b0;
        mem_write   = 1'b0;
        mem_read    = 1'b0;
        mem_to_reg  = 1'b0;
        branch      = 1'b0;
        alu_control = 4'b0000;
      end

      default: begin
        reg_write   = 1'b0;
        imm_src     = 1'b0;
        alu_src     = 1'b0;
        mem_write   = 1'b0;
        mem_read    = 1'b0;
        mem_to_reg  = 1'b0;
        branch      = 1'b0;
        alu_control = 4'b0000;
      end
    endcase
  end
endmodule





module scmarv32_top (
    input wire clk,
    input wire rst
);
  wire [31:0] pc_current;
  wire [31:0] pc_next;
  wire [31:0] pc_plus4;
  wire [31:0] pc_branch;
  wire [31:0] instr;

  wire reg_write;
  wire imm_src;
  wire alu_src;
  wire mem_write;
  wire mem_read;
  wire mem_to_reg;
  wire branch;
  wire jal;
  wire [3:0] alu_control;

  wire [31:0] rd1;
  wire [31:0] rd2;
  wire [31:0] imm_ext;
  wire [31:0] alu_src2;
  wire [31:0] alu_result;
  wire alu_zero;
  wire [31:0] read_data;
  wire [31:0] write_back_data;

  pc pc_unit (
      .clk(clk),
      .rst(rst),
      .pc_next(pc_next),
      .pc(pc_current)
  );

  pc_adder increment_adder (
      .pc(pc_current),
      .pc_next(pc_plus4)
  );

  instruction_reg imem (
      .A(pc_current),
      .instruction(instr)
  );

  control_unit ctrl (
      .opcode(instr[6:0]),
      .funct3(instr[14:12]),
      .funct7(instr[31:25]),
      .reg_write(reg_write),
      .imm_src(imm_src),
      .alu_src(alu_src),
      .mem_write(mem_write),
      .mem_read(mem_read),
      .mem_to_reg(mem_to_reg),
      .branch(branch),
      .jal(jal),
      .alu_control(alu_control)
  );

  reg_file rf (
      .clk(clk),
      .rst(rst),
      .reg_write(reg_write),
      .rs1(instr[19:15]),
      .rs2(instr[24:20]),
      .rd(instr[11:7]),
      .wd(write_back_data),
      .rd1(rd1),
      .rd2(rd2)
  );

  imm_gen imggen (
      .instruction(instr),
      .imm_ext(imm_ext)
  );

  assign alu_src2 = (alu_src) ? imm_ext : rd2;

  alu alu_unit (
      .A(rd1),
      .B(alu_src2),
      .alu_control(alu_control),
      .alu_output(alu_result),
      .zero(alu_zero)
  );

  data_mem dmem (
      .clk(clk),
      .rst(rst),
      .mem_read(mem_read),
      .mem_write(mem_write),
      .addr(alu_result),
      .write_data(rd2),
      .read_data(read_data)
  );

  assign write_back_data = (mem_to_reg) ? read_data : alu_result;

  assign pc_branch = $signed(pc_current) + $signed(imm_ext);

  wire take_branch = branch && ((instr[12] == 1'b0) ? alu_zero : !alu_zero);

  assign pc_next = (jal) ? pc_branch : (take_branch) ? pc_branch : pc_plus4;

  assign write_back_data = jal ? pc_plus4 : mem_to_reg ? read_data : alu_result;
endmodule
