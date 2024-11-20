#pragma once
#include <fstream>
#include <string>
#include <iostream>
using namespace std;


template <typename scalar_t>
int cuda_read_array(scalar_t **arr, string file)
{
    ifstream input(file, ios::in | ios::binary);
    input.seekg(0, input.end);
    int length = input.tellg();
    input.seekg(0, input.beg);
    int count = length / sizeof(scalar_t);
    cudaMallocManaged(arr, count * sizeof(scalar_t));

    input.read((char *)*arr, length);
    input.close();
    return count;
}
