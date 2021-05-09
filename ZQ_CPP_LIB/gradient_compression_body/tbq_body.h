#ifndef _OPERATOR_TBQ_BODY_H
#define _OPERATOR_TBQ_BODY_H
#include <stdint.h>
#include <thrust/random.h>
#include <thrust/sort.h>                       //sort()
#include <thrust/execution_policy.h>           //thrust::device
#include <thrust/functional.h>                 //greater<float>
#include <thrust/copy.h>                       //copy_if
#include <thrust/iterator/counting_iterator.h> // counting_iterator
#include <thrust/transform.h>                  //trnasform
#include "../naive_random.hpp"
#include "../operate_memory/get_policy_general.h"
#include "../likely.h"

namespace zq_cpp_lib
{
namespace gradient_compression_body
{
using namespace zq_cpp_lib::operate_memory;


  struct compressr_write_to{
    uint8_t* in;
    float* out;
    float threshold;
    compressr_write_to(
      uint8_t* _in,
      float* _out,
      float _threshold
    ):in(_in),out(_out),threshold(_threshold){}
    __host__ __device__
    float operator()(const int32_t&i) const{
      int32_t input_index = i>>2;
      int32_t input_offset = i&3;
      uint8_t qval = (in[input_index] >> (input_offset<<1)) & 3;
      return (static_cast<float>(qval)-1)*threshold;
    }
  };
  struct compressr_add_to{
    uint8_t* in;
    float* out;
    float threshold;
    compressr_add_to(
      uint8_t* _in,
      float* _out,
      float _threshold
    ):in(_in),out(_out),threshold(_threshold){}
    __host__ __device__
    float operator()(const int32_t&i) const{
      int32_t input_index = i>>2;
      int32_t input_offset = i&3;
      uint8_t qval = (in[input_index] >> (input_offset<<1)) & 3;
      return (static_cast<float>(qval)-1)*threshold + out[i];
    }
  };
  
  struct compress{
    float *grad, *residual;
    float threshold;
    compress(
      float *b,
      float *c,
      float d
    ){
      grad = b;
      residual = c;
      threshold = d;
    }
    __host__ __device__
    uint8_t operator()(const int32_t&i) const{
      int32_t start = i<<2;
      uint8_t qval = 85;
      int32_t k,j;
      for (k = 0; k < 4; k++){
        j = k + start;
        residual[j] += grad[j];
        if (residual[j] >= threshold){
          qval ^= (3<<(k<<1));
          residual[j] -= threshold;
        }
        else if (residual[j] <= -threshold){
          qval ^= (1<<(k<<1));
          residual[j] += threshold;
        }
      }
      return qval;
    }
  };

template <typename policy_t>
int tbq_body(
    float* to_compress_float,
    int32_t to_compress_float_size,
    float* residual_float,
    int32_t residual_float_size,
    uint8_t* out_uint8_t,
    int32_t out_uint8_t_size,
    float threshold,
    policy_t policy,
    void *stream)
{
  thrust::counting_iterator<int32_t> index_sequence_begin(0);
  thrust::transform(
    policy,
    index_sequence_begin,
    index_sequence_begin + (to_compress_float_size >> 2), //  (x+3) >> 2 ; take ceil 
    out_uint8_t,
    compress(
      to_compress_float,
      residual_float,
      threshold
    )
  );
  uint8_t left = to_compress_float_size & 3; // to_compress_float_size % 4
  if (left){
    float left_grad[4];
    float left_resi[4];
    get_policy<policy_t>::memcpyOut(
      left_grad,
      to_compress_float+(to_compress_float_size-left),
      sizeof(float)*left,
      stream
    );
    get_policy<policy_t>::memcpyOut(
      left_resi,
      residual_float+(to_compress_float_size-left),
      sizeof(float)*left,
      stream
    );
    get_policy<policy_t>::streamSynchronize(stream);
    uint8_t qval = 85;
    for (auto j = 0; j < left; j++){
      left_resi[j] += left_grad[j];
      if (left_resi[j] >= threshold){
        qval ^= (3<<(j<<1));
        left_resi[j] -= threshold;
      }
      else if (left_resi[j] <= -threshold){
        qval ^= (1<<(j<<1));
        left_resi[j] += threshold;
      }
    }
    get_policy<policy_t>::memcpyIn(
      residual_float + (to_compress_float_size - left),
      left_resi,
      sizeof(float)*left,
      stream
    );
    get_policy<policy_t>::memcpyIn(
      out_uint8_t + (to_compress_float_size >> 2),
      &qval,
      sizeof(uint8_t),
      stream
    );
  }

  return 0;
}

template <typename policy_t>
int tbq_r_body(
    float* out_float,
    int32_t out_float_size,
    uint8_t* in_uint8_t,
    int32_t in_uint8_t_size,
    float threshold,
    int is_add_to,
    policy_t policy,
    void *stream)
{
  thrust::counting_iterator<int32_t> index_sequence_begin(0);
  if (is_add_to){
    thrust::transform(
      get_policy<policy_t>::get(stream),
      index_sequence_begin,
      index_sequence_begin + (out_float_size),
      out_float,
      compressr_add_to(
        in_uint8_t,
        out_float,
        threshold
      )
    );
  }
  else{
    thrust::transform(
      get_policy<policy_t>::get(stream),
      index_sequence_begin,
      index_sequence_begin + (out_float_size),
      out_float,
      compressr_write_to(
        in_uint8_t,
        out_float,
        threshold
      )
    );
  }

  return 0;
}

} // namespace gradient_compression_body
} // namespace zq_cpp_lib

#endif
