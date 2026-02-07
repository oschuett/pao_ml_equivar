.SUFFIXES: .inp .out

inputs=$(shell echo */*42.inp)
outputs=$(addsuffix .out, $(basename $(inputs)))

all: $(outputs)

%.out: %.inp
	cd $(dir $<); OMP_NUM_THREADS=2 ../../../cp2k/build/bin/cp2k.ssmp $(notdir $<)  2>&1 > $(notdir $*.out)

#EOF
