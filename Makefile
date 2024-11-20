NVCC = nvcc
CFLAGS = -std=c++17 -arch=compute_75


EXEC1 = maxk_forward
SRC1 = main.cu spmm_maxk.cu spmm_cusparse.cu
LIB1 = -lcusparse -lstdc++fs

EXEC2 = max_pool_1d
SRC2 = max_pool_1d.cu


all: $(EXEC1) $(EXEC2)


$(EXEC1): $(SRC1)
	$(NVCC) $(CFLAGS) -o $(EXEC1) $(SRC1) $(LIB1)

$(EXEC2): $(SRC2)
	$(NVCC) $(CFLAGS) -o $(EXEC2) $(SRC2)

clean:
	rm -f $(EXEC1) $(EXEC2)

