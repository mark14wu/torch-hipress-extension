#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include "cuda_runtime.h"

#include "../time_cost.hpp"
#include "../gradient_compression_body/tbq_body.h"
#include "../operate_memory/get_policy_general.h"
#include "math.h"

int main(int argc, char**argv){
    zq_cpp_lib::time_cost zt;
    zt.start();
    int N = atoi(argv[1]);
    N = 1 << N;
    int M = (N+15)/16;
    float* gradient;
    float* residual;
    float* compressed;
    float* host_data;

    printf("N=%d\tM=%d\n",N, M);

    curandGenerator_t gen;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
    zt.record("create stream");

    host_data = (float*)calloc(N,sizeof(float));
    zt.record("calloc");

    cudaMalloc((void**)&gradient, N*sizeof(float));
    cudaMalloc((void**)&residual, N*sizeof(float));
    cudaMalloc((void**)&compressed, M*sizeof(float));
    zt.record("cudaMalloc");

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    zt.record("curand initialize");
    curandGenerateUniform(gen, gradient, N);
    zt.record("generate random values");

    cudaMemcpy(host_data, gradient, N*sizeof(float), cudaMemcpyDeviceToHost);
    zt.record("cudaMemcpy");
    
    printf("DATA BEFORE:\t");
    for (auto i = 0; i < 16; i++){
        printf("%1.4f ", host_data[i]);
    }
    printf("\n");
    zt.record("print original data");

    float threshold = 0.5;

    int ret = zq_cpp_lib::gradient_compression_body::tbq_body<thrust::cuda_cub::par_t::stream_attachment_type>(
        gradient,
        N,
        residual,
        N,
        reinterpret_cast<uint8_t*>(compressed),
        M*(sizeof(float)/sizeof(uint8_t)),
        threshold,
        policy,
        stream
    );
    if (ret) printf("ret=%d\n", ret);
    zt.record("tbq_body");

    ret = zq_cpp_lib::gradient_compression_body::tbq_r_body<thrust::cuda_cub::par_t::stream_attachment_type>(
        gradient,
        N,
        reinterpret_cast<uint8_t*>(compressed),
        M*(sizeof(float)/sizeof(uint8_t)),
        threshold,
        0,
        policy,
        stream
    );
    if (ret) printf("ret=%d\n", ret);
    zt.record("tbq_r_body");

    cudaMemcpy(host_data, gradient, N*sizeof(float), cudaMemcpyDeviceToHost);
    zt.record("cudaMemcpy");
    
    printf("DATA AFTER:\t");
    for (auto i = 0; i < 16; i++){
        printf("%1.4f ", host_data[i]);
    }
    printf("\n");
    zt.record("print decompressed data");

    curandDestroyGenerator(gen);
    cudaFree(gradient);
    cudaFree(compressed);
    cudaFree(residual);
    free(host_data);
    zt.record("destroy");
    zt.print_by_us();

    return 0;
}