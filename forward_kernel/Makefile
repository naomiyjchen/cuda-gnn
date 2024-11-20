NVCC = nvcc
CFLAGS = -std=c++17 -arch=compute_75


EXEC1 = forward_kernel
SRC1 = main.cu spmm_maxk.cu spmm_cusparse.cu
LIB1 = -lcusparse -lstdc++fs


all: $(EXEC1) 


$(EXEC1): $(SRC1)
	$(NVCC) $(CFLAGS) -o $(EXEC1) $(SRC1) $(LIB1)


clean:
	rm -f $(EXEC1) 

