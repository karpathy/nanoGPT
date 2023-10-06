//FPU Processing Element with flexible bits
import DEFINE_PKG::*;
module PE 
(
    input clk,
    input rstn,
    input [`BIT_W-1:0] in_l,
    input [`BIT_W-1:0] psum_in,
    input [`BIT_W-1:0] weights_in,
    output logic [`BIT_W-1:0] psum_out,
    output logic [`BIT_W-1:0] out_r
);
logic [`BIT_W-1:0] weights;
logic [`BIT_W-1:0] mult_out;
logic [`BIT_W-1:0] mac_out;
always_ff @( posedge clk or negedge rstn ) begin
    if(!rstn)
        weights<='0;
    else
        weights<=weights_in;
end
always_ff @( posedge clk or negedge rstn ) begin
    if(!rstn) begin
        psum_out<='0;
        out_r<='0;
    end
    else begin
        psum_out<=mac_out;
        out_r<=in_l;
    end
end

fmul mul (.a_in(weights),.b_in(in_l),.result(mult_out));
fadd add (.a_operand(mult_out), .b_operand(psum_in), .result(mac_out));
endmodule