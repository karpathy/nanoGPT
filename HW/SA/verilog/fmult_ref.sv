
// 
// Create Date: 2022/01/01 21:11:43
// Design Name: 
// Module Name: mul_f_sme425
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fmul_sme425(
      input aclk,                                  // input ire aclk
      input s_axis_a_tvalid,            // input ire s_axis_a_tvalid
      input [31:0] s_axis_a_tdata,              // input ire [31 : 0] s_axis_a_tdata
      input s_axis_b_tvalid,            // input ire s_axis_b_tvalid
      input [31:0] s_axis_b_tdata,              // input ire [31 : 0] s_axis_b_tdata
      output reg m_axis_result_tvalid,  // input ire m_axis_result_tvalid
      output [31:0] m_axis_result_tdata    // input ire [31 : 0] m_axis_result_tdata
    );
    
    wire [47:0] mul_fix_out;
    mul_fix_24 mul_fix(
    .clk(aclk),
    .a({1'b1, s_axis_a_tdata[22:0]}),
    .b({1'b1, s_axis_b_tdata[22:0]}),
    .c(mul_fix_out)
    );
    
    reg [31:0] a0, b0, a1, b1;
    always @(posedge aclk) begin
        a0 <= s_axis_a_tdata;
        b0 <= s_axis_b_tdata;
        a1 <= a0;
        b1 <= b0;
    end
    
    //zero check
    reg zero_check0, zero_check1;
    always @(posedge aclk) begin
        zero_check1 <= zero_check0;
        if (s_axis_a_tdata[30:23] == 8'd0 | s_axis_b_tdata[30:23] == 8'd0) begin
            zero_check0 <= 1'b1;
        end else zero_check0 <= 1'b0;
    end
    
    //generate M
    reg [22:0] M_result;
    always @(*) begin
        case(mul_fix_out[47:46])
            2'b01: M_result = mul_fix_out[45:23];
            2'b10: M_result = mul_fix_out[46:24];
            2'b11: M_result = mul_fix_out[46:24];
            default :M_result = mul_fix_out[46:24];
        endcase
    end
    
    //overflow check
    reg [8:0] e_result0;
    wire [7:0] e_result;
    always @(*) begin
        if (~zero_check1) begin
            if ((({1'b0, a1[30:23]} + {1'b0, b1[30:23]} + {8'd0, mul_fix_out[47]}) < 9'd127 | ({1'b0, a1[30:23]} + {1'b0, b1[30:23]} + {8'd0, mul_fix_out[47]}) > 9'd381))e_result0 = 9'h1ff; 
            else e_result0 = ({1'b0, a1[30:23]} + {1'b0, b1[30:23]} + {8'd0, mul_fix_out[47]}) - 9'd127;
        end
        else e_result0 = 9'd0;
    end
    assign e_result = e_result0[7:0];
    
    //sign
    wire sign;
    assign sign = a1[31] ^ b1[31];
    
    //result_valid
    reg result_valid;
    always @(posedge aclk) begin
        if (s_axis_a_tvalid & s_axis_b_tvalid) result_valid <= 1'b1;
        else result_valid <= 1'b0;
        m_axis_result_tvalid <= result_valid;
    end
    
    wire [22:0] overflow;
    assign overflow = (zero_check1 || ({1'b0, a1[30:23]} + {1'b0, b1[30:23]} + {8'd0, mul_fix_out[47]}) < 9'd127 || ({1'b0, a1[30:23]} + {1'b0, b1[30:23]} + {8'd0, mul_fix_out[47]}) > 9'd381)?23'd0:23'hffffff;
    
    assign m_axis_result_tdata = {sign, e_result, overflow & M_result};
endmodule