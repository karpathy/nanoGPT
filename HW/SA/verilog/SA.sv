import DEFINE_PKG::*;
module SA (
    input clk,
    input rstn,
    input [`DIMENSION-1:0][`BIT_W-1:0] input_top,
    input [`DIMENSION-1:0][`BIT_W-1:0] input_left,
    input [`DIMENSION-1:0][`DIMENSION-1:0][`BIT_W-1:0] weights_in, //for testing convenience.
    output logic [`DIMENSION-1:0] [`BIT_W-1:0] out_bot,
    output logic [`DIMENSION-1:0] [`BIT_W-1:0] out_right
);

logic [`BIT_W-1:0] hori_wires [`DIMENSION-1:0][`DIMENSION-2:0];// horizaton wire: dimension * (dimension - 1)
logic [`BIT_W-1:0] ver_wires [`DIMENSION-2:0][`DIMENSION-1:0];// vertical wire: (dimension - 1) * dimension




generate
   for (genvar i = 0; i < `DIMENSION; ++i)begin : gen_sa_row
      for (genvar j = 0; j < `DIMENSION; ++j)begin : gen_sa_col
         if (i == 0)begin
            if (j == 0)begin //SA[0][0]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(input_left[i]), .psum_in(input_top[j]),
                  .psum_out(ver_wires[i][j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
            end else if (j == `DIMENSION-1) begin //SA[0][N-1]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(hori_wires[i][j-1]), .psum_in(input_top[j]),
                  .psum_out(ver_wires[i][j]), .out_r(out_right[i]), .weights_in(weights_in[i][j])
               );
            end else begin //SA[0][1:N-2]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(hori_wires[i][j-1]), .psum_in(input_top[j]),
                  .psum_out(ver_wires[i][j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
            end      
         end else if (i == `DIMENSION-1) begin
               if (j == 0)begin //SA[N-1][0]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(input_left[i]), .psum_in(ver_wires[i-1][j]),
                  .psum_out(out_bot[j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
            end else if (j == `DIMENSION-1) begin //SA[N-1][N-1]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(hori_wires[i][j-1]), .psum_in(ver_wires[i-1][j]),
                  .psum_out(out_bot[j]), .out_r(out_right[i]), .weights_in(weights_in[i][j])
               );
            end else begin //SA[N-1][1:N-2]
               PE  PE_ij (
                  .clk(clk), .rstn(rstn),
                  .in_l(hori_wires[i][j-1]), .psum_in(ver_wires[i-1][j]),
                  .psum_out(out_bot[j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
            end                  
         end else if (j == 0) begin //SA[1:N-2][0]
               PE  PE_ij (
               .clk(clk), .rstn(rstn),
               .in_l(input_left[i]), .psum_in(ver_wires[i-1][j]),
               .psum_out(ver_wires[i][j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
         end else if (j == `DIMENSION-1) begin //SA[1:N-2][N-1]
               PE  PE_ij (
               .clk(clk), .rstn(rstn),
               .in_l(hori_wires[i][j-1]), .psum_in(ver_wires[i-1][j]),
               .psum_out(ver_wires[i][j]), .out_r(out_right[i]), .weights_in(weights_in[i][j])
               );
         end else begin //SA[1][1:N-2:N-2]
               PE  PE_ij (
               .clk(clk), .rstn(rstn),
               .in_l(hori_wires[i][j-1]), .psum_in(ver_wires[i-1][j]),
               .psum_out(ver_wires[i][j]), .out_r(hori_wires[i][j]), .weights_in(weights_in[i][j])
               );
         end      
         
      end
   end
endgenerate
endmodule