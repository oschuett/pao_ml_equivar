.SUFFIXES: .inp .out

inputs=$(shell echo Frame*/*.inp)
outputs=$(addsuffix .out, $(basename $(inputs)))

all: $(outputs)

%.out: %.inp
	cd $(dir $<); mpiexec -np 32 ../../../cp2k/exe/local/cp2k.popt $(notdir $<)  2>&1 > $(notdir $*.out)

#EOF
