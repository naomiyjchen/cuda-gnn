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
//const int N = 716847 >> 8 << 8;
//const int N = 2449029 >> 8 << 8;
//const int N = 16;

const int dim_in = 256, dim_out = 192;

__global__ void maxpool(float *, float *, unsigned int *);

int main() {

    cout << "Max Pooling kernel" << endl;
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

    cout << "data ready, testing..." << endl;

    unsigned int shared_mem_size = WARPS_PER_BLOCK * dim_in * sizeof(float);

    cout<<"Config GridDim = "<< N / WARPS_PER_BLOCK << ", BlockDim = " << WARPS_PER_BLOCK * 32 << ", shared_mem_size = " << shared_mem_size << endl;

    int times = 100;
    for (int i = 0; i < times; i++) {
        maxpool <<< N / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32 >>> (data, value, indices);
    }

    cudaDeviceSynchronize();
    double measured_time = 0;

    for (int i = 0; i < times; i++) {
        timestamp(t0);
        maxpool <<< N / WARPS_PER_BLOCK, WARPS_PER_BLOCK * 32 >>> (data, value, indices);
        cudaDeviceSynchronize();
        timestamp(t1);
        measured_time += getDuration(t0, t1);
    }

    cout << "max-pooling time = " << measured_time / times * 1000 << " ms" <<endl;

//    for (int i = 0; i < dim_out; i += 1) {
//        cout << "value[" << i << "] = " << *(value + i) << endl;
//    }
//
//    for (int i = 0; i < dim_out; i += 1) {
//        cout << "indices[" << i << "] = " << *(indices + i) << endl;
//    }

    cudaFree(data);
    cudaFree(value);
    cudaFree(indices);

    return 0;
}

__global__ void maxpool(float *data, float *value, unsigned int *indices) {

    const int warp_id = threadIdx.x / 32;
    const int local_tid = threadIdx.x % 32;
    const int warp_offset = WARPS_PER_BLOCK * dim_in;
    const int vertex_offset = blockIdx.x * warp_offset + warp_id * dim_in;
    const int sqrt_dim_in = 16;

    int xx = local_tid / 4 * 2;
    int yy = local_tid % 4 * 4;

    unsigned int pos;
    float v;

#pragma unroll
    for (unsigned int i = 0; i < 2; i += 1) {

        yy += 2 * i;

        pos = xx * sqrt_dim_in + yy;
        v = data[vertex_offset + pos];

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + xx * sqrt_dim_in + yy + 1] > v) {
            pos = xx * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy + 1] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        data[vertex_offset + pos] = -1.0;

        value[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i] = v;
        indices[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i] = pos;
    }

    yy = local_tid % 4 * 4;

#pragma unroll
    for (unsigned int i = 0; i < 2; i += 1) {

        yy += 2 * i;

        pos = xx * sqrt_dim_in + yy;
        v = data[vertex_offset + pos];

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + xx * sqrt_dim_in + yy + 1] > v) {
            pos = xx * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy + 1] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        data[vertex_offset + pos] = -1.0;

        value[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i + 2] = v;
        indices[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i + 2] = pos;
    }

    yy = local_tid % 4 * 4;

#pragma unroll
    for (unsigned int i = 0; i < 2; i += 1) {

        yy += 2 * i;

        pos = xx * sqrt_dim_in + yy;
        v = data[vertex_offset + pos];

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + xx * sqrt_dim_in + yy + 1] > v) {
            pos = xx * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        if (data[vertex_offset + (xx + 1) * sqrt_dim_in + yy + 1] > v) {
            pos = (xx + 1) * sqrt_dim_in + yy + 1;
            v = data[vertex_offset + pos];
        }

        value[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i + 4] = v;
        indices[blockIdx.x * WARPS_PER_BLOCK * dim_out + warp_id * dim_out + 6 * local_tid + i + 4] = pos;
    }
}