CUDA_COMPILER = nvcc
CXX_FLAGS = -std=c++17
CUDA_ARCH = -gencode arch=compute_75,code=sm_75
CUDA_FLAGS = -std=c++17 $(CUDA_ARCH)


EXEC1 = maxk_forward
SRC1 = main.cu spmm_maxk.cu spmm_cusparse.cu
LIBRARIES = -lcusparse -lstdc++fs

EXEC2 = max_pool_1d
SRC2 = max_pool_1d.cu

all: $(EXEC1) $(EXEC2)


$(EXEC1): $(SRC1)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CXX_FLAGS) -o $@ $(SRC1) $(LIBRARIES)

$(EXEC2): $(SRC2)
	$(CUDA_COMPILER) $(CUDA_FLAGS) $(CXX_FLAGS) -o $(EXEC2) $(SRC2)
# Clean target
clean:
	rm -f $(EXEC1) $(EXEC2)

