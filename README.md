# Final Project

Author: Pang-Li Yang (py2236), You-Jun Chen (yc7093)

The experiment was done on NYU's GPU Computing CUDA2 cluster, running Red Hat Enterprise Linux 9. The system is equipped with two Intel Xeon E5-2660 CPUs (40 cores total), 256 GB of RAM, and two GeForce RTX 2080 Ti GPU cards (each with 11 GB of VRAM, CUDA version 12.4). We wrote our kernels in C++ and CUDA C++ and compiled our source code using g++ 11.5 and nvcc 12.4, targeting compute architecture 7.5.

**To compile the forward kernel:**

```
cd forward_kernel
make
./forward_kernel
```

We have included the warp-level partitioned graphs. Optionally, you can use the below command to generate them yourself.

```
cd forward_kernel
python generate_meta.py
```

**To compile the topk approximator kernels**

```
cd topk_approximator_kernels
make
./maxpool_1d
./maxpool_2d
./topk
```
