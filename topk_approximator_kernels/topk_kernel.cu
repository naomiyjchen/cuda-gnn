#include <iostream>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

using namespace std;

const int WARPS_PER_BLOCK = 16;
const int N = 232960 >> 8 << 8;
//const int N = 4096;

const int dim_in = 256, dim_out = 64;

__global__ void topk(float *, float *, unsigned int *);

int main() {

    cout << "Test TopK kernel" << endl;
    cout << "N = "<< N << ", dim_in = " << dim_in << ", dim_out = " << dim_out << ", preparing data..." << endl;

    float *data, *value;
    unsigned int *indices;

    cudaMallocManaged(&data,  N * dim_in  * sizeof(float));
    cudaMallocManaged(&value, N * dim_out * sizeof(float));
    cudaMallocManaged(&indices, N * dim_out * sizeof(unsigned int));

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    generate(data, data + N * dim_in, [&](){ return rd(engine); });

    unsigned int shared_mem_size = WARPS_PER_BLOCK * dim_in * (sizeof(float) + sizeof(unsigned int));

    cout<<"Config GridDim = "<< N / WARPS_PER_BLOCK << ", BlockDim = " << WARPS_PER_BLOCK * 32 << ", shared_mem_size = " << shared_mem_size << endl;

    dim3 grid(N / WARPS_PER_BLOCK, 1, 1);
    dim3 block(WARPS_PER_BLOCK * 32, 1, 1);

    int times = 10;
    for (int i = 0; i < times; i++) {
        topk <<< grid, block, shared_mem_size >>> (data, value, indices);
    }

    cudaDeviceSynchronize();
    double measured_time = 0;

    for (int i = 0; i < times; i++) {
        timestamp(t0);
        topk <<< grid, block, shared_mem_size >>> (data, value, indices);
        cudaDeviceSynchronize();
        timestamp(t1);
        measured_time += getDuration(t0, t1);
    }

    cout << "top-k time = " << measured_time / times * 1000 << " ms" <<endl;

    for (int i = 0; i < 64; i += 1) {
        cout << "value[" << i << "] = " << *(value + i) << endl;
    }

    for (int i = 0; i < 64; i += 1) {
        cout << "indices[" << i << "] = " << *(indices + i) << endl;
    }

    cudaFree(data);
    cudaFree(value);
    cudaFree(indices);

    return 0;
}

__global__ void topk(float *data, float *value, unsigned int *indices) {

    extern __shared__ float buffer[];
    unsigned int *track = (unsigned int*) &buffer[WARPS_PER_BLOCK * dim_in];

    const int warp_id = threadIdx.x / 32;
    const int local_tid = threadIdx.x % 32;
    const int warp_offset = WARPS_PER_BLOCK * dim_in;
    const int feature_per_warp = dim_in / 32;

    float v_holder;
    unsigned int idx_holder;

    #pragma unroll
    for (unsigned int i = 0; i < feature_per_warp; i += 1) {
        buffer[warp_id * dim_in + feature_per_warp * local_tid + i] = data[blockIdx.x * warp_offset + warp_id * dim_in + feature_per_warp * local_tid + i];
        track[warp_id * dim_in + feature_per_warp * local_tid + i] = local_tid * feature_per_warp + i;
    }

    __syncwarp();

    #pragma unroll
    for (int iter = 0; iter < dim_in / 2; iter += 1) {

        for (int i = 0; i < dim_in; i += 64) {

            int curr_idx = warp_id * dim_in + 2 * local_tid + i;

            if (buffer[curr_idx] < buffer[curr_idx + 1]) {
                v_holder = buffer[curr_idx];
                buffer[curr_idx] = buffer[curr_idx + 1];
                buffer[curr_idx + 1] = v_holder;

                idx_holder = track[curr_idx];
                track[curr_idx] = track[curr_idx + 1];
                track[curr_idx + 1] = idx_holder;
            }

        }

        __syncwarp();

        for (int i = 0; i < dim_in; i += 64) {

            int curr_idx = warp_id * dim_in + 2 * local_tid + i + 1;

            if (curr_idx < dim_in - 1 && buffer[curr_idx] < buffer[curr_idx + 1]) {
                v_holder = buffer[curr_idx];
                buffer[curr_idx] = buffer[curr_idx + 1];
                buffer[curr_idx + 1] = v_holder;

                idx_holder = track[curr_idx];
                track[curr_idx] = track[curr_idx + 1];
                track[curr_idx + 1] = idx_holder;
            }
        }

        __syncwarp();
    }

    __syncwarp();

    for (int i = 0; i < dim_out; i += 32) {

        value[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + local_tid + i] = buffer[warp_id * dim_in + local_tid + i];
        indices[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + local_tid + i] = track[warp_id * dim_in + local_tid + i];
    }
    
}