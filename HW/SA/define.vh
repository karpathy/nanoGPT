`ifndef _DEFINE_SVH_
`define _DEFINE_SVH_ 
package DEFINE_PKG;

`define DIMENSION 4
`define M_W     23   
`define EXP_W   8
`define BIT_W   32
`define MULT_W  `M_W+`M_W+2
`define EXP_MAX  2**(`EXP_W-1)+2**(`EXP_W)-3

`define N_TESTS 100000

endpackage

`endif
