#include <iostream>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();
inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b) {
     return std::chrono::duration<double>(b - a).count();
}

const int WARPS_PER_BLOCK = 16;
const int WARP_SIZE = 32;
const int N = 232965 / 256 * 256;
const int dim_in = 256, dim_out = 64;
const int window_size = dim_in / dim_out;
const int windows_per_thread = (dim_out + WARP_SIZE - 1) / WARP_SIZE; 

using namespace std;



__global__ void max_pool_1d(float*, float*, int*);

void printValues(const float*, int, int, const string&);
void printIndices(const int*, int, int, const string&);


int main() {
    cout << "Max Pool 1D  kernel" << endl;
    cout<<"N = "<< N << ", dim_in = " << dim_in << ", dim_out = " << dim_out << ", preparing data..."<<endl;

    /*
     * Memory Allocation
     */

    float *data, *values;
    int *indices;

    cudaMallocManaged(&data, N * dim_in * sizeof(float));
    cudaMallocManaged(&values, N * dim_out * sizeof(float));
    cudaMallocManaged(&indices, N * dim_out * sizeof(int));

    /*
     * Data Initialization
     */ 
    
    default_random_engine engine;
    engine.seed(123);
    uniform_real_distribution<float> rd(0, 1);
    generate(data, data + N * dim_in, [&](){ return rd(engine); });
    
    cout << "data ready, testing..." << endl;

    int shared_mem_size = WARPS_PER_BLOCK * dim_in * sizeof(float);
    int grid_size = N / WARPS_PER_BLOCK;
    int block_size = WARPS_PER_BLOCK * WARP_SIZE;

	cout<<"Config GridDim = "<< N / WARPS_PER_BLOCK << ", BlockDim = " << WARPS_PER_BLOCK * 32 << ", shared_mem_size = " << shared_mem_size << endl;

    // warmp up
    int times = 100;
    for (int i = 0; i < times; i++) {
        max_pool_1d<<< grid_size, block_size, shared_mem_size>>>(data, values, indices);
    }
    cudaDeviceSynchronize();

    
    // testing
    double measured_time = 0;

    for (int i = 0; i < times; i++) {  
        timestamp(t0);
        max_pool_1d<<< grid_size, block_size, shared_mem_size>>>(data, values, indices);
        cudaDeviceSynchronize();
        
        timestamp(t1);
        measured_time += getDuration(t0, t1);
    }
    
    cout << "max pool 1d kernel time = " << measured_time / times * 1000 << " ms" << endl;


    // printValues(data, 1, dim_in, "Data (input)");
    // printValues(values, 1, dim_out, "Value (output)");
    // printIndices(indices, 1, dim_out, "indices (output)");
   
 
    cudaFree(data);
    cudaFree(values);
    cudaFree(indices);
         
    return 0;
}



__global__ void max_pool_1d(float *data, float *output_values, int *output_indices) {
    
    extern __shared__ float buffer[];
    float *node_data = buffer;

    int warp_id = threadIdx.x / WARP_SIZE;  // warp within block
    int lane_id = threadIdx.x % WARP_SIZE;  // thread within warp
    int node_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;


    // Load data into shared memory
    for (int i = lane_id; i < dim_in; i += WARP_SIZE) {
        node_data[warp_id * dim_in + i] = data[node_id * dim_in + i];
    }
    
    __syncwarp();


        

    #pragma unroll
    for (int w = 0; w < windows_per_thread; ++w) {
        int window_id = lane_id + w * WARP_SIZE;
        
        int window_start = warp_id * dim_in + window_id * window_size;
        float max_val = 0.0;
        int max_idx = -1;

        // Find max within each window assigned to this thread
        for (int i = 0; i < window_size; ++i) {
            if (node_data[window_start + i] > max_val) {
                max_val = node_data[window_start + i];
                max_idx = (window_start + i) % dim_in;
            }
        }
        
        output_values[node_id * dim_out + window_id] = max_val;
        output_indices[node_id * dim_out + window_id] = max_idx;
    }


}    



void printValues(const float* data, int N, int dim, const string& label) {
    cout << "\n" << label << ":" << endl;
    for (int i = 0; i < N; ++i) {
        cout << "Node " << i << ": " << endl;
        for (int j = 0; j < dim; ++j) {
            cout << fixed << setprecision(3) << data[i * dim + j] << " ";
        }
        cout << endl;
    }
}


void printIndices(const int* data, int N, int dim, const string& label) {
    cout << "\n" << label << ":" << endl;
    for (int i = 0; i < N; ++i) {
        cout << "Node: " << i << ": " << endl;
        for (int j = 0; j < dim; ++j) {
            cout << setw(3) << (int)data[i * dim + j] << " ";
        }
        cout << endl;
    }
}
