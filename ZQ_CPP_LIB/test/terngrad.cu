#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <curand.h>
#include "cuda_runtime.h"

#include "../time_cost.hpp"
#include "../gradient_compression_body/terngrad_body.h"
#include "../operate_memory/get_policy_general.h"
#include "math.h"

int main(int argc, char **argv)
{

    zq_cpp_lib::time_cost zt;
    zt.start();
    int N = atoi(argv[1]);
    N = 1 << N;
    int M = 10 + (N + 3) / 4;
    printf("N=%d\n", N);
    printf("M=%d\n", M);

    float *devData;
    float *hostData;
    uint8_t *COMPRESSED;
    curandGenerator_t gen;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    zt.record("create stream");

    auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
    zt.record("get policy");

    hostData = (float *)calloc(N, sizeof(float));
    zt.record("calloc");
    cudaMalloc((void **)&devData, N * sizeof(float));
    cudaMalloc((void **)&COMPRESSED, M * sizeof(uint8_t));
    zt.record("cudaMalloc");
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    zt.record("curand initialize");
    curandGenerateUniform(gen, devData, N);
    curandGenerateUniform(gen, devData, N);
    zt.record("generate random twice");
    cudaMemcpy(hostData, devData, N * sizeof(float), cudaMemcpyDeviceToHost);
    zt.record("cudaMemcpy");
    for (auto i = 0; i < 16; i++)
    {
        printf("%1.4f ", hostData[i]);
    }
    printf("\n");
    zt.record("print result");
    int ret = zq_cpp_lib::gradient_compression_body::terngrad_body<thrust::cuda_cub::par_t::stream_attachment_type>(
        devData,
        N,
        COMPRESSED,
        M,
        2,
        1,
        policy,
        stream);
    if (ret) printf("ret=%d\n", ret);
    zt.record("first call terngrad_body");
    for (auto i = 0; i < 10; i++){
        int ret = zq_cpp_lib::gradient_compression_body::terngrad_body<thrust::cuda_cub::par_t::stream_attachment_type>(
            devData,
            N,
            COMPRESSED,
            M,
            2,
            1,
            policy,
            stream);
        if (ret) printf("ret=%d\n", ret);
    }
    zt.record("call terngrad_body 10 times");

    curandDestroyGenerator(gen);
    cudaFree(devData);
    free(hostData);
    zt.record("destroy");
    zt.print_by_us();

    return 0;
}
