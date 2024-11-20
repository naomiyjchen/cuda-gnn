#include <iostream>
#include "data.h"
#include "spmm_maxk.h"
#include "spmm_cusparse.h"
#include <random>
#include <algorithm>
#include <filesystem>

string base_dir = "./graphs/";

int total_file_cnt, current_file_cnt;

using namespace std;

#define DIM_MUL_N 1
#define DIM_MUL(x) ((x + DIM_MUL_N - 1) / DIM_MUL_N) * DIM_MUL_N

double check_err(float *out, float *out_ref, int len, bool &has_err)
{
    double err_sum = 0;
    has_err = 0;

    for (int i = 0; i < len; i++)
    {
        double err = abs(out[i] - out_ref[i]);
        err_sum += err;
        
        if (err > 0.1)
        {
            has_err = 1;
            cout << "err at " << i << " err = " << err << " ref = " << out_ref[i] <<endl;
        }
    }
    cout << "err sum = " << err_sum << "  ";
    if (err_sum / len < 0.001)
    {
        cout << "validation pass!" << endl;
    }
    else
    {
        cout << "validation fail!" << endl;
    }
    return err_sum;
}

void test_graph(string graph)
{
    int dim_origin = 256;
    int dim_k_list[] = {32, 64, 96, 128, 192};
    int dim_k_limit = 192;

    int *csr_indptr, *csr_indices;
    int v_num = cuda_read_array(&csr_indptr, base_dir + graph + ".indptr") - 1;
    int e_num = cuda_read_array(&csr_indices, base_dir + graph + ".indices");

    float *csr_value;
    cudaMallocManaged(&csr_value, e_num * sizeof(float));

    float *vout_cusparse;
    float *vin_sparse, *vin_sparse_data, *vout_maxk;
    u_int8_t *vin_sparse_selector;
    cudaMallocManaged(&vout_cusparse, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&vin_sparse, v_num * dim_origin * sizeof(float));
    cudaMallocManaged(&vin_sparse_data, v_num * DIM_MUL(dim_k_limit) * sizeof(float));
    cudaMallocManaged(&vin_sparse_selector, v_num * DIM_MUL(dim_k_limit) * sizeof(u_int8_t));
    cudaMallocManaged(&vout_maxk, v_num * dim_origin * sizeof(float));
    

    default_random_engine engine;
    engine.seed(123);

    uniform_real_distribution<float> rd(0, 1);

    generate(csr_value, csr_value + e_num, [&]() { return rd(engine); });
    generate(vin_sparse_data, vin_sparse_data + v_num * dim_k_limit, [&]() { return rd(engine); });
    generate(vin_sparse, vin_sparse + v_num * dim_origin, [&]() { return rd(engine); });

    vector<int> sequence(dim_origin);
    iota(sequence.begin(), sequence.end(), 0); 

    SPMM_MAXK maxk_forward(graph, csr_indptr, csr_indices, csr_value, vin_sparse_data, vout_maxk, v_num, e_num, dim_origin);
    
    double t_cusparse;
    cout << "num graph dim_origin dim_k kernel time(ms)" << endl;

    for (int n = 0; n < sizeof(dim_k_list) / sizeof(int); n++)
    {
        int dim_k = dim_k_list[n];
        if(dim_k > dim_k_limit){
            break;
        }

        string outstr = to_string(current_file_cnt) + "/" + to_string(total_file_cnt) + " " + graph + " " + to_string(dim_origin) + " " + to_string(dim_k);

        vector<int> sample(dim_k);

        for (int i = 0; i < v_num; ++i)
        {
            std::sample(sequence.begin(), sequence.end(), sample.begin(), dim_k, engine);

            for (int j = 0; j < dim_k; ++j)
            {
                float v = rd(engine);
                vin_sparse_data[i * DIM_MUL(dim_k) + j] = v;
                vin_sparse_selector[i * DIM_MUL(dim_k) + j] = sample[j];
            }
        }

        for (int i = 0; i < v_num; ++i)
        {
            for (int j = 0; j < dim_origin; ++j)
            {
                vin_sparse[i * dim_origin + j] = 0.0;
            }
            for (int j = 0; j < dim_k; ++j)
            {
                int col = vin_sparse_selector[i * DIM_MUL(dim_k) + j];
                vin_sparse[i * dim_origin + col] = vin_sparse_data[i * DIM_MUL(dim_k) + j];
            }
        }

        maxk_forward.vin_sparse_selector = vin_sparse_selector;
        maxk_forward.dim_sparse = dim_k;


        if(n == 0){
        	cout << outstr << endl;
        	spmm_cusparse(csr_indptr, csr_indices, csr_value, vin_sparse, vout_cusparse, v_num, e_num, dim_origin, 0);
        	maxk_forward.do_test(false, dim_origin);
        	bool has_err = 0;
        	check_err(vout_maxk, vout_cusparse, v_num * dim_origin, has_err);
        }

        t_cusparse = spmm_cusparse(csr_indptr, csr_indices, csr_value, vin_sparse, vout_cusparse, v_num, e_num, dim_origin, 10);
        cout << outstr << " cusparse " << t_cusparse * 1000 << endl;

        double t_maxk = maxk_forward.do_test(true, dim_origin);
        cout << outstr << " maxk_forward " << t_maxk * 1000 << ", speed up: " << t_cusparse / t_maxk << endl;
    }

    cudaFree(csr_indptr);
    cudaFree(csr_indices);
    cudaFree(csr_value);
    cudaFree(vout_cusparse);
    cudaFree(vin_sparse);
    cudaFree(vin_sparse_data);
    cudaFree(vin_sparse_selector);
    cudaFree(vout_maxk);
}

int main(int argc, char *argv[])
{
    if (argc > 1)
    {
        string arg_graph(argv[1]);
        test_graph(arg_graph);
    }
    else
    {
        string folder_path = base_dir;
        string extension = ".indptr";

        total_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                total_file_cnt++;
            }
        }

        current_file_cnt = 0;
        for (const auto &file : filesystem::directory_iterator(folder_path))
        {
            if (file.path().extension() == extension)
            {
                current_file_cnt++;
                string graph = file.path().stem().string();
                test_graph(graph);
                cudaDeviceSynchronize();
            }
        }
    }

    return 0;
}
