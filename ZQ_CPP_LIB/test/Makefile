CU_NEEDED=$(subst .cu,_nvcc.o,$(wildcard *.cu))
CC_NEEDED=$(subst .cc,_gpp.o,$(wildcard *.cc))

ALL: CU CC

CU: $(CU_NEEDED)
CC: $(CC_NEEDED)




NVCC=/usr/local/cuda/bin/nvcc
GPP=g++
COMMON_FLAGS=--std=c++11 -I../../
NVCC_FLAGS=-lcurand

%_gpp.o: %.cc
	$(GPP) $< -o $@ $(COMMON_FLAGS)
%_nvcc.o: %.cu
	$(NVCC) $< -o $@ $(COMMON_FLAGS) $(NVCC_FLAGS)
clean:
	rm -rf $(CU_NEEDED) $(CC_NEEDED)


