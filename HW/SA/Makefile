STD_CELLS = /afs/umich.edu/class/eecs627/tsmc180/sc-x_2004q3v1/aci/sc/verilog/tsmc18_neg.v
#TESTBENCH = ../scan/tb/clock_gen.sv ../scan/tb/tb.v ../scan/tb/scan_test.v
TESTBENCH = ../SA/tb/tb.sv
G3_PATH = /afs/umich.edu/class/eecs627/w23/groups/group3
# ../TOP/verilog/define.vh
# ../SA_array/verilog/SA_Array.sv
#remember to change sram_rtl.v to sram.v when doing post apr sim
SIM_FILES = ../SA/define.vh ../SA/verilog/fadd.sv ../SA/verilog/fmul.sv ../SA/verilog/PE.sv ../SA/verilog/SA.sv
SIM_SYNTH_FILES = ../TOP/verilog/define.vh
SIM_APR_FILES = /afs/umich.edu/class/eecs627/w23/groups/group3/TOP/verilog/define.vh ../TSMC_Controller/apr/controller.apr.v ../TSMC_Shifter/apr/shifter.apr.v \
				../TSMC_write_back/apr/write_back_part.apr.v ../TSMC_SA_array_part/apr/SA_array_part.apr.v $(G3_PATH)/TSMC_SRAM/sram_apr17/compout/views/sram/Typical/sram.v \
				../TSMC_integration_antenna/apr/data/TOP_nodiode.apr.v /afs/umich.edu/class/eecs627/w23/groups/group3/virtuoso_example/digital/Ring_full/apr/Ring_full.apr.v
VV         = vcs
VVOPTS     = +v2k +vc -sverilog -timescale=1ns/1ps +vcs+lic+wait +multisource_int_delays  +lint=TFIPC-L                   \
	       	+neg_tchk +incdir+$(VERIF) +plusarg_save +overlap +warn=noSDFCOM_UHICD,noSDFCOM_IWSBA,noSDFCOM_IANE,noSDFCOM_PONF -full64 -cc gcc +libext+.v+.vlib+.vh 

ifdef WAVES
VVOPTS += +define+DUMP_VCD=1 +memcbk +vcs+dumparrays +sdfverbose
endif

ifdef GUI
VVOPTS += -gui
endif

all: clean c_compile sim synth sim_synth sim_apr

clean:
	rm -f ucli.key
	rm -f sim
	rm -f sim_synth
	rm -fr sim.daidir
	rm -fr sim_synth.daidir
	rm -rf *.log
	rm -fr csrc
	rm -rf -r syn/dwsvf_*
	rm -f syn/mult.syn.v
	rm -f syn/output.txt
	rm -f syn/*.{log,sdf,rpt,svf}
	rm -f inter.*
	rm -f *afs*
	rm -f novas.*
	rm -rf sim*
	rm -f *.txt
	rm -rf sim_apr
	rm -rf sdf/*.sdf

c_compile:
	cd goldenbrick; gcc -Wall -ggdb -o goldenbrick goldenbrick.c
	cd goldenbrick; ./goldenbrick > goldenbrick.txt

sim: clean
	$(VV) -o $@  $(VVOPTS) -debug_access+all $(SIM_FILES) $(TESTBENCH) -kdb -R -gui | tee sim_result.txt
# $(VV) -o $@  $(VVOPTS) -debug_access+all $(SIM_FILES) $(TESTBENCH); ./$@

dve:	$(SIM_FILES)  $(TESTBENCH)
#	$(VV) $(VVOPTS) -lncurses $(SIM_FILES) $(TESTBENCH) -debug_access+all -kdb -o verilog/$@ -R -gui
	$(VV) $(VVOPTS) -lncurses $^ -debug_access+all -kdb -o $@ -R -gui

synth:
	cd syn; dc_shell -tcl_mode -xg_mode -f mult_8bit.syn.tcl | tee output.txt 

sim_synth:
	cd verilog; $(VV) $(VVOPTS) $(STD_CELLS) $(SIM_SYNTH_FILES) $(TESTBENCH) -kdb -R -gui | tee output.txt

run_apr:
	cd apr; mv BitBrick_8bit.apr.tcl viewDefinition.tcl BitBrick.globals BitBrick_8bit.io ../; rm -rf ./*; cd ..; mv BitBrick_8bit.apr.tcl viewDefinition.tcl BitBrick.globals BitBrick_8bit.io apr/
	cd apr; innovus -init BitBrick_8bit.apr.tcl | tee output.txt 

copy_from_apr:
	cp $(G3_PATH)/TSMC_SA_array_part/apr/SA_array_part.apr.sdf $(G3_PATH)/TOP/sdf
	cp $(G3_PATH)/TSMC_write_back/apr/write_back_part.apr.sdf $(G3_PATH)/TOP/sdf
	cp $(G3_PATH)/TSMC_Controller/apr/controller.apr.sdf $(G3_PATH)/TOP/sdf
	cp $(G3_PATH)/TSMC_Shifter/apr/shifter.apr.sdf $(G3_PATH)/TOP/sdf
	cp /afs/umich.edu/class/eecs627/w23/groups/group3/virtuoso_example/digital/Ring_full/apr/Ring_full.apr.sdf $(G3_PATH)/TOP/sdf
	cp $(G3_PATH)/TSMC_integration_antenna/apr/data/TOP.apr.sdf $(G3_PATH)/TOP/sdf

sim_apr: clean copy_from_apr
	$(VV) $(VVOPTS) +sdfverbose -debug_access+all +define+APR=1 $(STD_CELLS) $(SIM_APR_FILES) $(TESTBENCH) -kdb -R -gui | tee output.txt
#	-o $@ ; ./$@

sim_apr_new: clean copy_from_apr
	@ $(VV) $(VVOPTS) +sdfverbose -debug_access+all +define+APR=1 $(STD_CELLS) $(SIM_APR_FILES) $(TESTBENCH)
	@ ./simv;

verdi_apr:
	@ verdi -sverilog -2012 -autoalias +define_FSDB_DUMP +define+APR=1 $(STD_CELLS) $(SIM_APR_FILES) $(TESTBENCH) -top testbench -ssf ./TB.fsdb & 