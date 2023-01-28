.SUFFIXES: .inp .out

inputs=$(shell echo frame_*/*_pao44.inp)
outputs=$(addsuffix .out, $(basename $(inputs)))

all: $(outputs)

%.out: %.inp
	cd $(dir $<); ../../../cp2k/exe/local/cp2k.sopt $(notdir $<)  2>&1 > $(notdir $*.out)

#EOF
