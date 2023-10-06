import DEFINE_PKG::*;
module fmul(
      input [`BIT_W-1:0] a_in,
      input [`BIT_W-1:0] b_in,
      output [`BIT_W-1:0] result
    );
    
    logic [`MULT_W-1:0] mul_fix_out;
    // mul_fix mult(
    // .clk(aclk),
    // .a({1'b1, a_in[`M_W-1:0]}),
    // .b({1'b1, b_in[`M_W-1:0]}),
    // .c(mul_fix_out)
    // );
    assign mul_fix_out={1'b1, a_in[`M_W-1:0]}*{1'b1, b_in[`M_W-1:0]};
    
    //zero check
    logic zero_check;
    always_comb begin
        if (a_in[`BIT_W-2:`M_W] == '0 | b_in[`BIT_W-2:`M_W] == '0) begin
            zero_check = 1'b1;
        end else zero_check = 1'b0;
    end
    
    //generate M
    logic [`M_W-1:0] M_result;
    always_comb begin
        case(mul_fix_out[`MULT_W-1:`MULT_W-2])
            2'b01: M_result = mul_fix_out[`MULT_W-3:`M_W];
            2'b10: M_result = mul_fix_out[`MULT_W-2:`M_W+1];
            2'b11: M_result = mul_fix_out[`MULT_W-2:`M_W+1];
            default :M_result = mul_fix_out[`MULT_W-2:`M_W+1];
        endcase
    end
    
    //overflow check
    logic [`EXP_W:0] e_result0;
    logic [`EXP_W-1:0] e_result;
    logic overflow;

    assign overflow= (zero_check || ({1'b0, a_in[`BIT_W-2:`M_W]} + {1'b0, b_in[`BIT_W-2:`M_W]} + {{`EXP_W{1'b0}}, mul_fix_out[`MULT_W-1]}) < {2'b0,{(`EXP_W-1){1'b1}}} || ({1'b0, a_in[`BIT_W-2:`M_W]} + {1'b0, b_in[`BIT_W-2:`M_W]} + {8'd0, mul_fix_out[`MULT_W-1]}) > `EXP_MAX);
    
    always_comb begin
        if (~zero_check) begin
            //exp1+exp2+msb_of_mult_out
            if (overflow)e_result0 = {(`EXP_W+1){1'b1}}; 
            else e_result0 = ({1'b0, a_in[`BIT_W-2:`M_W]} + {1'b0, b_in[`BIT_W-2:`M_W]} + {{`EXP_W{1'b0}}, mul_fix_out[`MULT_W-1]}) - {2'b0,{(`EXP_W-1){1'b1}}};
        end
        else e_result0 = '0;
    end
    assign e_result = e_result0[`EXP_W-1:0];
    
    //sign
    logic sign;
    assign sign = a_in[`BIT_W-1] ^ b_in[`BIT_W-1];
    
    logic [`M_W-1:0] overflow_mask;
    assign overflow_mask = overflow ?'0:{(`M_W){1'b1}};
    
    assign result = {sign, e_result, overflow_mask & M_result};
endmodule