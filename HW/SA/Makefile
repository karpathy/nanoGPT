TESTBENCH = ../SA/tb/tb.sv
SIM_FILES = ../SA/define.vh ../SA/verilog/fadd.sv ../SA/verilog/fmul.sv ../SA/verilog/PE.sv ../SA/verilog/SA.sv

VV         = vcs
VVOPTS     = +v2k +vc -sverilog -timescale=1ns/1ps +vcs+lic+wait +multisource_int_delays  +lint=TFIPC-L                   \
	       	+neg_tchk +incdir+$(VERIF) +plusarg_save +overlap +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF -full64 -cc gcc +libext+.v+.vlib+.vh 

ifdef WAVES
VVOPTS += +define+DUMP_VCD=1 +memcbk +vcs+dumparrays +sdfverbose
endif

ifdef GUI
VVOPTS += -gui
endif

all: clean sim

clean:
	rm -f ucli.key
	rm -f sim
	rm -f sim_synth
	rm -fr sim.daidir
	rm -rf *.log
	rm -fr csrc

sim: clean
	$(VV) -o $@  $(VVOPTS) -debug_access+all $(SIM_FILES) $(TESTBENCH) -kdb -R -gui | tee sim_result.txt

dve:	$(SIM_FILES)  $(TESTBENCH)
	$(VV) $(VVOPTS) -lncurses $^ -debug_access+all -kdb -o $@ -R -gui