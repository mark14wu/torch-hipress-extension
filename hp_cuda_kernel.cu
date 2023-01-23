#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <stdint.h>
#include <thrust/random.h>
#include <thrust/sort.h>                       //sort()
#include <thrust/execution_policy.h>           //thrust::device
#include <thrust/functional.h>                 //greater<float>
#include <thrust/copy.h>                       //copy_if
#include <thrust/iterator/counting_iterator.h> // counting_iterator
#include <thrust/transform.h>                  //trnasform
#include "./ZQ_CPP_LIB/operate_memory/get_policy_general.h"
#include "./ZQ_CPP_LIB/gradient_compression_body/terngrad_body.h"
#include "./ZQ_CPP_LIB/gradient_compression_body/tbq_body.h"
#include "./ZQ_CPP_LIB/gradient_compression_body/gd_body.h"
#include "./ZQ_CPP_LIB/gradient_compression_body/powersgd_body.h"
#include "./ZQ_CPP_LIB/likely.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#define PVOID2CUDASTREAM(stream) (static_cast<cudaStream_t>(stream))




void hp_cuda_terngrad(
  std::vector<float*>& in_floats,
  std::vector<int32_t>& in_float_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<int32_t>& out_uint8_t_sizes,
  std::vector<uint8_t>& bitwidths,
  std::vector<int32_t>& enable_randoms,
  cudaStream_t stream
){
  //static int flag = 0;
  //static cudaStream_t stream;
  //static thrust::cuda_cub::par_t::stream_attachment_type policy;
  //if (!flag){
  //  stream = c10::cuda::getCurrentCUDAStream();
  //  std::cout << "stream: " << stream << std::endl;
  //  policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  //  flag = 1;
  //}

  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  
  
  int ret = zq_cpp_lib::gradient_compression_body::terngrad_body<thrust::cuda_cub::par_t::stream_attachment_type>(
    in_floats,
    in_float_sizes,
    out_uint8_ts,
    out_uint8_t_sizes,
    bitwidths,
    enable_randoms,
    policy,
    stream
  );
  
  if (ret) printf("ret=%d\n",ret);
  // return output;
}

void hp_cuda_gd(
  std::vector<float*>& in_floats,
  std::vector<int32_t>& Ns,
  std::vector<float*>& residuals,
  std::vector<int32_t>& residual_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<double>& sample_rates,
  std::vector<double>& drop_ratios,
  cudaStream_t stream
){
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  for (uint i = 0; i < in_floats.size(); i++){
    int ret = zq_cpp_lib::gradient_compression_body::gd_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      in_floats[i],
      Ns[i],
      residuals[i],
      residual_sizes[i],
      out_uint8_ts[i],
      sample_rates[i],
      drop_ratios[i],
      policy,
      stream
    );
    if (unlikely(ret)){
      std::cout << "Call gd_body failed, ret = " << ret << "." << std::endl;
    }
  }
}

void hp_cuda_gdr(
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
){
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  
  int ret = zq_cpp_lib::gradient_compression_body::gd_r_body<thrust::cuda_cub::par_t::stream_attachment_type>(
    in_uint8_ts,
    in_uint8_t_sizes,
    out_floats,
    out_float_sizes,
    is_add_tos,
    policy,
    stream
  );
  if (unlikely(ret)){
    std::cout << "Call gdr_body failed, ret = " << ret << "." << std::endl;
  }
}

void hp_cuda_tbq(
  std::vector<float*>& to_compress_floats,
  std::vector<int32_t>& to_compress_float_sizes,
  std::vector<float*>& residual_floats,
  std::vector<int32_t>& residual_float_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<int32_t>& out_uint8_t_sizes,
  std::vector<float>& thresholds,
  cudaStream_t stream
){
  //static int flag = 0;
  //static cudaStream_t stream;
  //static thrust::cuda_cub::par_t::stream_attachment_type policy;
  //if (!flag){
  //  stream = c10::cuda::getCurrentCUDAStream();
  //  std::cout<< "stream:" << stream << std::endl;
  //  policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  //  flag = 1;
  //}
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);

  for (uint i = 0; i < to_compress_floats.size(); i++){
    int ret = zq_cpp_lib::gradient_compression_body::tbq_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      to_compress_floats[i],
      to_compress_float_sizes[i],
      residual_floats[i],
      residual_float_sizes[i],
      out_uint8_ts[i],
      out_uint8_t_sizes[i],
      thresholds[i],
      policy,
      stream
    );
    if (unlikely(ret)){
      std::cout << "Call tbq_body failed, ret = " << ret << "." << std::endl;
    }
  }
  cudaStreamSynchronize(PVOID2CUDASTREAM(stream));
}


void hp_cuda_tbqr(
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<float>& thresholds,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
){
  //static int flag = 0;
  //static cudaStream_t stream;
  //static thrust::cuda_cub::par_t::stream_attachment_type policy;
  //if (!flag){
  //  stream = c10::cuda::getCurrentCUDAStream();
  //  std::cout<< "stream:" << stream << std::endl;
  //  policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  //  flag = 1;
  //}

  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  
  for (uint i = 0; i < out_floats.size(); i++){
    int ret = zq_cpp_lib::gradient_compression_body::tbq_r_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      out_floats[i],
      out_float_sizes[i],
      in_uint8_ts[i],
      in_uint8_t_sizes[i],
      thresholds[i],
      is_add_tos[i],
      policy,
      stream
    );
    if (unlikely(ret)){
      std::cout << "Call tbq_body failed, ret = " << ret << "." << std::endl;
    }
  }
  cudaStreamSynchronize(PVOID2CUDASTREAM(stream));
}

void hp_cuda_terngradr(
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
){
  //static int flag = 0;
  //static cudaStream_t stream;
  //static thrust::cuda_cub::par_t::stream_attachment_type policy;
  //if (!flag){
  //  stream = c10::cuda::getCurrentCUDAStream();
  //  std::cout<< "stream:" << stream << std::endl;
  //  policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  //  flag = 1;
  //}

  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  
  int ret = zq_cpp_lib::gradient_compression_body::terngrad_r_body<thrust::cuda_cub::par_t::stream_attachment_type>(
    out_floats,
    out_float_sizes,
    in_uint8_ts,
    in_uint8_t_sizes,
    is_add_tos,
    policy,
    stream
  );
  if (ret) printf("ret=%d\n", ret);
}

void hp_cuda_powersgd_encode1(
  std::vector<std::shared_ptr<torch::Tensor>>& grads,
  std::vector<std::shared_ptr<torch::Tensor>>& residuals,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  c10::cuda::CUDAStream& stream
) {
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  for (uint i = 0; i < grads.size(); i++) {
    int ret = zq_cpp_lib::gradient_compression_body::powersgd_encode1_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      *grads[i],
      *residuals[i],
      *Qs[i],
      *Ms[i],
      *Ps[i],
      policy,
      stream
    );
    if (unlikely(ret)) {
      std::cout << "Call powersgd encode1 failed, ret = " << ret << "." << std::endl;
    }
  }
}

void hp_cuda_powersgd_encode2(
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  c10::cuda::CUDAStream& stream
) {
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  for (uint i = 0; i < Ps.size(); i++) {
    int ret = zq_cpp_lib::gradient_compression_body::powersgd_encode2_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      *Ps[i],
      *Ms[i],
      *Qs[i],
      policy,
      stream
    );
    if (unlikely(ret)) {
      std::cout << "Call powersgd encode1 failed, ret = " << ret << "." << std::endl;
    }
  }
}

void hp_cuda_powersgd_decode(
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& residuals,
  std::vector<std::shared_ptr<torch::Tensor>>& grads,
  c10::cuda::CUDAStream& stream
) {
  auto policy = zq_cpp_lib::operate_memory::get_policy<thrust::cuda_cub::par_t::stream_attachment_type>::get(stream);
  for (uint i = 0; i < Ps.size(); i++) {
    int ret = zq_cpp_lib::gradient_compression_body::powersgd_decode_body<thrust::cuda_cub::par_t::stream_attachment_type>(
      *Ps[i],
      *Qs[i],
      *Ms[i],
      *residuals[i],
      *grads[i],
      policy,
      stream
    );
    if (unlikely(ret)) {
      std::cout << "Call powersgd encode1 failed, ret = " << ret << "." << std::endl;
    }
  }
}

