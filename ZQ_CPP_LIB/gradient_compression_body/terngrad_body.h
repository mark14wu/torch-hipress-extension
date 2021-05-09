#ifndef _ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_TERNGRAD_BODY_H
#define _ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_TERNGRAD_BODY_H
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
#include <vector>
#include <chrono>

namespace zq_cpp_lib
{
namespace gradient_compression_body
{
using namespace zq_cpp_lib::operate_memory;


struct compress_without_random{
  float* in_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_val;
  float gap_inverse;
  compress_without_random(
    float* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_float = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_val = d;
    gap_inverse = e;
  }
  __host__ __device__
  uint8_t operator()(const int32_t& i){
    uint8_t qval = 0;
    int j;
    float thetimes;
    uint8_t t;
#pragma unroll
    for (j = 0; j < (1<<data_per_byte_lg2); j++){
      thetimes = (in_float[(i<<data_per_byte_lg2) + j] - min_val) * gap_inverse;
      t = nearbyint(thetimes);
      qval |= (t << (bitwidth*j));
    };
    return qval;
  }
};


struct compress_with_random{
  float* in_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_val;
  float gap_inverse;
  unsigned long long timestamp;
  compress_with_random(
    float* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e,
    unsigned long long f
  ){
    in_float = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_val = d;
    gap_inverse = e;
    timestamp = f;
  }
  __host__ __device__
  uint8_t operator()(const int32_t& i){
    uint8_t qval = 0;
    int j;
    float thetimes;
    uint8_t t;
    zq_cpp_lib::naive_real_random<float> r(0.0,1.0);
    r.srand(timestamp+i);
#pragma unroll
    for (j = 0; j < (1<<data_per_byte_lg2); j++){
      thetimes = (in_float[(i<<data_per_byte_lg2) + j] - min_val) * gap_inverse;
      thetimes += r();
      t = static_cast<uint8_t>(thetimes);
      qval |= (t << (bitwidth*j));
    };
    return qval;
  }
};

struct decompress_write_to{
  uint8_t* in_uint8_t;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_f;
  float gap;
  decompress_write_to(
    uint8_t* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_uint8_t = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_f = d;
    gap = e;
  }
  __host__ __device__
  float operator()(const int32_t& i){
    int32_t input_index = (i >> data_per_byte_lg2) + 10;
    uint8_t input_offset = i & ((1 << data_per_byte_lg2) - 1);
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (in_uint8_t[input_index] >> (input_offset * bitwidth)) & mask;
    return static_cast<float>(qval*gap + min_f);
  }
};
struct decompress_add_to{
  uint8_t* in_uint8_t;
  float* out_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_f;
  float gap;
  decompress_add_to(
    uint8_t* a,
    float* out_float_,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_uint8_t = a;
    out_float = out_float_;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_f = d;
    gap = e;
  }
  __host__ __device__
  float operator()(const int32_t& i){
    int32_t input_index = (i >> data_per_byte_lg2) + 10;
    uint8_t input_offset = i & ((1 << data_per_byte_lg2) - 1);
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (in_uint8_t[input_index] >> (input_offset * bitwidth)) & mask;
    return static_cast<float>(qval*gap + min_f + out_float[i]);
  }
};


template <typename policy_t>
int terngrad_body(
    std::vector<float*> in_floats,
    std::vector<int32_t> in_float_sizes,
    std::vector<uint8_t*> out_uint8_ts,
    std::vector<int32_t> out_uint8_t_sizes,
    std::vector<uint8_t> bitwidths,
    std::vector<int32_t> randoms,
    policy_t policy,
    void *stream)
{

  int comp_size = in_floats.size();
  if(comp_size <= 0){
    return 0;
  }

  std::vector<std::shared_ptr<float>> min_vals;
  std::vector<std::shared_ptr<float>> max_vals;

  for(int i = 0; i < comp_size; i++){
    float min_val, max_val;
    min_vals.push_back(std::make_shared<float>(min_val));
    max_vals.push_back(std::make_shared<float>(max_val));
    auto min_max = thrust::minmax_element(
      policy,
      in_floats.at(i),
      in_floats.at(i)+in_float_sizes.at(i)
    );
    get_policy<policy_t>::memcpyOut(&(*min_vals.at(i)),min_max.first,sizeof(float),stream);
    get_policy<policy_t>::memcpyOut(&(*max_vals.at(i)),min_max.second,sizeof(float),stream);
  }

  get_policy<policy_t>::streamSynchronize(stream);


  std::vector<float> gap_inverses;
  std::vector<uint8_t> data_per_byte_lg2s;
  std::vector<uint8_t> data_per_bytes;
  std::vector<uint8_t> tails;
  uint8_t lg2[9] = {0,0,1,1,2,2,2,2,3};

  for(int i = 0; i < comp_size; i++){
    float gap = (*(max_vals.at(i)) - *(min_vals.at(i))) / ((1 << bitwidths.at(i)) - 1.0f);
    float gap_inverse = 1. / (gap + 1e-8);
    uint8_t bitwidth_lg2 = lg2[bitwidths.at(i)];

    if (unlikely((1<<bitwidth_lg2)!=bitwidths.at(i))){
      printf("Invalid value of bitwidth, chekc value: bitwidth=%d\n",bitwidths.at(i)+0);
      return -1;
    }

    uint8_t data_per_byte_lg2 = 3 - bitwidth_lg2;
    uint8_t data_per_byte = 1<<data_per_byte_lg2;
    uint8_t tail = in_float_sizes.at(i) % data_per_byte;
    tail = tail ==  0 ? 0 : data_per_byte - tail; 

    uint8_t header[10];
    ((float*)(header+2))[0] = *(min_vals.at(i));
    ((float*)(header+6))[0] = *(max_vals.at(i));
    header[0] = bitwidths.at(i);
    header[1] = tail;
    get_policy<policy_t>::memcpyIn(out_uint8_ts.at(i), header, sizeof(uint8_t)*10, stream);

    gap_inverses.push_back(gap_inverse);
    data_per_byte_lg2s.push_back(data_per_byte_lg2);
    data_per_bytes.push_back(data_per_byte);
    tails.push_back(tail);
  }


  for(int i = 0; i < comp_size; i++){
    thrust::counting_iterator<int32_t> index_sequence_begin(0);
    if (randoms.at(i)){
      thrust::transform(
        policy,
        index_sequence_begin,
        index_sequence_begin + (in_float_sizes.at(i) >> data_per_byte_lg2s.at(i)),
        out_uint8_ts.at(i)+10,
        compress_with_random(
          in_floats.at(i),
          bitwidths.at(i),
          data_per_byte_lg2s.at(i),
          *(min_vals.at(i)),
          gap_inverses.at(i),
          static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now()
            .time_since_epoch()
            .count()
          )
        )
      );
    }
    else{
      thrust::transform(
        policy,
        index_sequence_begin,
        index_sequence_begin + (in_float_sizes.at(i) >> data_per_byte_lg2s.at(i)),
        out_uint8_ts.at(i)+10,
        compress_without_random(
          in_floats.at(i),
          bitwidths.at(i),
          data_per_byte_lg2s.at(i),
          *(min_vals.at(i)),
          gap_inverses.at(i)
        )
      );
    }
  }

  std::vector<float*> tail_datas;
  

  for(int i = 0; i < comp_size; i++){
    float* tail_data = new float[8];
    if (tails.at(i)){
      // cudaMemcpy(tail_data,in_float+in_float_size-data_per_byte,sizeof(float)*(data_per_byte-tail),cudaMemcpyDeviceToHost);
      // in_float_size - data_per_byte?
      get_policy<policy_t>::memcpyOut(tail_data,in_floats.at(i)+(in_float_sizes.at(i)-(data_per_bytes.at(i)-tails.at(i))),sizeof(float)*(data_per_bytes.at(i)-tails.at(i)),stream);
    } 
    tail_datas.push_back(tail_data);
  }

  get_policy<policy_t>::streamSynchronize(stream);
  
  for(int i = 0; i < comp_size; i++){
    if (tails.at(i)){
      uint8_t qval = 0;
      auto tail_data = tail_datas.at(i);   
      for (auto j = 0; j < data_per_bytes.at(i) - tails.at(i); j++){
        uint8_t t = nearbyint((tail_data[j] - *(min_vals.at(i)))*gap_inverses.at(i));
        qval = qval | ( t << (bitwidths.at(i)*j));
      };
      // cudaMemcpyAsync(out_uint8_t+out_uint8_t_size-1,&qval,sizeof(uint8_t),cudaMemcpyHostToDevice,mshadow::Stream<gpu>::GetStream(s));
      get_policy<policy_t>::memcpyIn(out_uint8_ts.at(i)+(out_uint8_t_sizes.at(i)-1), &qval, sizeof(uint8_t), stream);
    };
  }

  //free memory space
  for(auto tail_data : tail_datas){
    delete tail_data;
  }
   
  return 0;

}

template <typename policy_t>
int terngrad_r_body(
    std::vector<float*> out_floats,
    std::vector<int32_t> out_float_sizes,
    std::vector<uint8_t*> in_uint8_ts,
    std::vector<int32_t> in_uint8_t_sizes,
    std::vector<int> is_add_tos,
    policy_t policy,
    void *stream)
{

  int decomp_size = in_uint8_ts.size();
  if (decomp_size <= 0){
    return 0;
  }

  std::vector<uint8_t> bitwidths;
  std::vector<uint8_t> bitwidth_lg2s;
  std::vector<float> min_vals;
  std::vector<float> gaps;
  std::vector<uint8_t> tails;
  std::vector<uint8_t> data_per_byte_lg2s;
  
  std::vector<uint8_t*> headers;

  for(int i = 0; i < decomp_size; i++){
    uint8_t* header = new uint8_t[10];
    get_policy<policy_t>::memcpyOut(header, in_uint8_ts.at(i), 10*sizeof(uint8_t),stream);
    headers.push_back(header);
  }

  get_policy<policy_t>::streamSynchronize(stream);


  uint8_t lg2[9] = {0,0,1,1,2,2,2,2,3};

  for(int i = 0; i < decomp_size; i++){
    auto header = headers.at(i);
    float min_val = *((float*)(header+2));
    float max_val = *((float*)(header+6));
    uint8_t bitwidth = header[0];
    uint8_t tail = header[1];
    float gap = (max_val - min_val) / ((1 << bitwidth) - 1.0f);
    uint8_t bitwidth_lg2 = lg2[bitwidth];

    if (unlikely((1<<bitwidth_lg2) != bitwidth)){
      printf("bitwidth is invalid, check value: %d\n", bitwidth+0);
      return -1;
    } else {
      bitwidths.push_back(bitwidth);
      bitwidth_lg2s.push_back(bitwidth_lg2);
      min_vals.push_back(min_val);
      gaps.push_back(gap);
      data_per_byte_lg2s.push_back(3-bitwidth_lg2);
      tails.push_back(tail);
    }
  }


  for(int i = 0; i < decomp_size; i++){
    thrust::counting_iterator<int32_t> index_sequence_begin(0);
    if (is_add_tos.at(i)){
      thrust::transform(
        policy,
        index_sequence_begin,
        index_sequence_begin + (((in_uint8_t_sizes.at(i)-10)<<data_per_byte_lg2s.at(i))-tails.at(i)),
        out_floats.at(i),
        decompress_add_to(
          in_uint8_ts.at(i),
          out_floats.at(i),
          bitwidths.at(i),
          data_per_byte_lg2s.at(i),
          min_vals.at(i),
          gaps.at(i)
        )
      );
    }
    else{
      thrust::transform(
        policy,
        index_sequence_begin,
        index_sequence_begin + (((in_uint8_t_sizes.at(i)-10) << data_per_byte_lg2s.at(i)) - tails.at(i)),
        out_floats.at(i),
        decompress_write_to(
          in_uint8_ts.at(i),
          bitwidths.at(i),
          data_per_byte_lg2s.at(i),
          min_vals.at(i),
          gaps.at(i)
        )
      );
    }
  }
  
  //free memory space
  for(auto header : headers){
    delete header;
  }

  return 0;
}

} // namespace gradient_compression_body
} // namespace zq_cpp_lib

#endif
