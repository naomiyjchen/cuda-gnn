NVCC = nvcc
CFLAGS = -arch=compute_75

EXEC1 = maxpool_1d
EXEC2 = maxpool_2d
EXEC3 = topk

SRC1 = maxpool_1d_kernel.cu
SRC2 = maxpool_2d_kernel.cu
#SRC2 = maxpool_noShared.cu
SRC3 = topk_kernel.cu

all: $(EXEC1) $(EXEC2) $(EXEC3)

$(EXEC1): $(SRC1)
	$(NVCC) -o $(EXEC1) $(SRC1) $(CFLAGS)

$(EXEC2): $(SRC2)
	$(NVCC) -o $(EXEC2) $(SRC2) $(CFLAGS)

$(EXEC3): $(SRC3)
	$(NVCC) -o $(EXEC3) $(SRC3) $(CFLAGS)

clean:
	rm -f $(EXEC1) $(EXEC2) $(EXEC3)
