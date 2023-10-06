import DEFINE_PKG::*;


module tb;
    logic clk=0;
    logic rstn;
    logic [`DIMENSION-1:0][`BIT_W-1:0] input_top;
    logic [`DIMENSION-1:0][`BIT_W-1:0] input_left;
    logic [`DIMENSION-1:0][`DIMENSION-1:0][`BIT_W-1:0] weights_in; //for testing convenience.
    logic [`DIMENSION-1:0] [`BIT_W-1:0] out_bot;
    logic [`DIMENSION-1:0] [`BIT_W-1:0] out_right;
	

	SA DUT(.*);

	always #5 clk = ~clk;


    initial begin
        rstn=0;
        for(int i=0;i<`DIMENSION;i++) begin
            input_top[i]=0;
            input_left[i]=32'h4ac25fb5;
            for(int j=0;j<`DIMENSION;j++) begin
                weights_in[i][j]=32'hcb057dd0;
            end

        end
        @(negedge clk)
        @(negedge clk)
        rstn=1;
        @(negedge clk)
        for(int i=0;i<`DIMENSION;i++) begin
            input_left[i]=32'h499c2468;
        end
        @(negedge clk)
        for(int i=0;i<`DIMENSION;i++) begin
            input_left[i]=32'h4abf5cf8;
        end
        @(negedge clk)
        for(int i=0;i<`DIMENSION;i++) begin
            input_left[i]=32'hc919ca88;
        end
        
        #10000
        $stop;


    end

endmodule