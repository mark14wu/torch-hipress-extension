#ifndef ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_ECQ_BODY_H
#define ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_ECQ_BODY_H
#include <stdint.h>
#include <thrust/copy.h> //copy_if
#include <thrust/execution_policy.h> //thrust::device
#include <thrust/functional.h> //greater<float>
#include <thrust/iterator/counting_iterator.h> // counting_iterator
#include <thrust/random.h>
#include <thrust/sort.h> //sort()
#include <thrust/transform.h> //trnasform
#include "../likely.h"
#include "../naive_random.hpp"
#include "../operate_memory/get_policy_general.h"

namespace zq_cpp_lib {
namespace gradient_compression_body {
using namespace zq_cpp_lib::operate_memory;


struct generate_max_GH_index{
  float* G;
  float* H;
  double alpha;
  generate_max_GH_index(
    float* G_,
    float* H_,
    double alpha_
  ):
  G(G_),
  H(H_),
  alpha(alpha_)
  {}
  __host__ __device__
  int32_t operator()(const int32_t&x, const int32_t&y){
    return fabs(G[x] + alpha*H[x]) > fabs(G[y] + alpha*H[y])
      ? x
      : y;
  }
};

struct generate_GH{
  float* G;
  float* H;
  float* GH;
  double alpha;
  generate_GH(
    float* G_,
    float* H_,
    float* GH_,
    double alpha_
  ):
  G(G_),
  H(H_),
  GH(GH_),
  alpha(alpha_)
  {}
  __host__ __device__
  void operator()(const int32_t&x){
      // GH[x] = fabs(G[x] + h);
      GH[x] = fabs(G[x] + alpha*H[x]);
      // H[x] *= alpha; //4.4 -> 6.0ms
      // H[x] /= 2; //4.4->5.9ms
  }
};

struct generate_Q_update_H{
  float* GH;
  float* H;
  uint8_t* Q;
  double alpha;
  double beta;
  int32_t N;
  float max_GH;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float gap;
  float gap_inverse;
  uint64_t t;
  generate_Q_update_H(
    float* GH_,
    float* H_,
    uint8_t* Q_,
    double alpha_,
    double beta_,
    int32_t N_,
    float max_GH_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float gap_,
    float gap_inverse_,
    uint64_t t_
  ):
  GH(GH_),
  H(H_),
  Q(Q_),
  alpha(alpha_),
  beta(beta_),
  N(N_),
  max_GH(max_GH_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  gap(gap_),
  gap_inverse(gap_inverse_),
  t(t_)
  {}
  __host__ __device__
  void operator()(const int32_t x){
    uint8_t q = 0;
    
    int32_t head = x * data_per_byte;
    int32_t tail = head+data_per_byte;
    float f;
    int32_t j;
    zq_cpp_lib::naive_real_random<float>r;
    r.srand(t+x);
    #pragma unroll
    for (j=head; j < tail; j++){
      f = (GH[j] + max_GH)*gap_inverse;
      uint8_t t = static_cast<uint8_t>(f+r());
      q = q | (t << (bitwidth*(j-head)));
      H[j] = beta*H[j] + (f-t)*gap; //7.8ms-5ms
      // H[j] = beta*H[j] + (f-t); //7.8ms-5ms
      // H[j] = (f-t)*gap; //5.2ms-5ms
      // float g_ = t * gap - max_GH;
      // H[j] = beta*H[j] + G[j] - g_; //9ms - 5ms
      // H[j] = max_GH;//5ms-5ms
      // H[j] = beta*H[j] + G[j] - max_GH; //8.5ms - 5ms 
      // H[j] = beta*H[j]; //7.2ms - 5ms 
      // H[j] = 1+H[j]; //5.3ms - 5ms 
    }
    Q[x] = q;
  }
};
struct generate_Q_with_0{
  uint8_t* Q;
  generate_Q_with_0(
    uint8_t* Q_
  ):Q(Q_){};
  __host__ __device__
  void operator()(const int32_t& x){
    Q[x] = 0;
  }
};
struct update_H_with_0{
  float* H;
  double beta;
  update_H_with_0(
    float* H_,
    double beta_
  ): H(H_), beta(beta_){};
  __host__ __device__
  void operator()(const int32_t& x){
    H[x] = H[x] * beta;
  }
};

template <typename policy_t>
int ecq_body(
    float* G,
    int32_t G_size,
    float* H,
    int32_t H_size,
    uint8_t* out_uint8_t,
    int32_t out_uint8_t_size,
    double alpha,
    double beta,
    int32_t bitwidth,
    policy_t policy,
    void* stream) {
  
  uint8_t* Q = out_uint8_t+10;
  if (unlikely(bitwidth!=2 && bitwidth!=4 && bitwidth!=8)){
    printf("bitwidth's range: {2,4,8}\tInput bitwidth=%d\n", bitwidth+0);
    CHECK_EQ(0,1);
  }
  uint8_t s = (1<<(bitwidth-1)) - 1; // (1<<bitwidth - 2) / 2
  uint8_t data_per_byte = 8 / bitwidth;
  int32_t N = G_size;
  int32_t M = static_cast<int32_t>(ceil(N*1.0/data_per_byte));
  float* GH = reinterpret_cast<float*>(out_uint8_t);

  // t_comment[t_index] = "initialize";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    generate_GH(
      G,
      H,
      GH,
      alpha
    )
  );
  // t_comment[t_index] = "generate_GH";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  float* max_GH_p = thrust::max_element(
    policy,
    GH,
    GH+N
  );
  float max_GH;
  get_policy<policy_t>::memcpyOut(
    &max_GH,
    max_GH_p,
    sizeof(float),
    stream
  );
  // t_comment[t_index] = "find max element in GH";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();

  int32_t out_true_size = 4+4+1+1+M;
  
  if (unlikely(max_GH < 1e-6)){
    // printf("max_GH is 0! Please check your data!");
    // CHECK_EQ(0,1);
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(M-1),
      generate_Q_with_0(
        Q
      )
    );
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      update_H_with_0(
        H,
        beta
      )
    );
  }
  else{
    float gap = max_GH / s;
    float gap_inverse = 1.0 / gap;

    // printf("data_per_byte=%d\n",data_per_byte+0);
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(M-1),
      generate_Q_update_H(
        GH,
        H,
        Q,
        alpha,
        beta,
        N,
        max_GH,
        data_per_byte,
        bitwidth,
        gap,
        gap_inverse,
        std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count()
      )
    );
  }

  // get_policy<policy_t>::streamSynchronize(stream);
  // t_comment[t_index] = "generate_Q_update_H";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  uint8_t header[10];
  memcpy(header,&out_true_size,sizeof(int32_t));
  memcpy(header+4, &max_GH, sizeof(float));
  header[8]=bitwidth;
  header[9]=static_cast<uint8_t>(M*data_per_byte-N);
  get_policy<policy_t>::memcpyIn(
    out_uint8_t,
    header,
    sizeof(uint8_t)*10,
    stream
  );
  // t_comment[t_index] = "copy several bytes";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  // printf("time cost:\t");
  // for (int32_t i = 0; i < t_index-1; i++){
  //   if (t_comment[i+1] != ""){
  //     printf("(%s)",t_comment[i+1].c_str());
  //   }
  //   auto t_cost = zq_cpp_lib::get_cost_time_by_us(t_list[i],t_list[i+1]);
  //   printf("%.0lf\t", t_cost);
  // }
  // printf("\n");
  return 0;
}


struct generate_G{
  uint8_t* Q;
  float* G;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float max_abs_G;
  float gap;
  generate_G(
    uint8_t* Q_,
    float* G_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float max_abs_G_,
    float gap_
  ):
  Q(Q_),
  G(G_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  max_abs_G(max_abs_G_),
  gap(gap_)
  {}
  __host__ __device__
  void operator()(const int32_t& x) {
    int32_t Q_index = x / data_per_byte;
    int32_t Q_offset = x % data_per_byte;
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (Q[Q_index] >> (Q_offset*bitwidth))&mask;
    G[x] = qval*gap - max_abs_G;
  }
};

struct generate_G_add_to{
  uint8_t* Q;
  float* G;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float max_abs_G;
  float gap;
  generate_G_add_to(
    uint8_t* Q_,
    float* G_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float max_abs_G_,
    float gap_
  ):
  Q(Q_),
  G(G_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  max_abs_G(max_abs_G_),
  gap(gap_)
  {}
  __host__ __device__
  void operator()(const int32_t& x) {
    int32_t Q_index = x / data_per_byte;
    int32_t Q_offset = x % data_per_byte;
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (Q[Q_index] >> (Q_offset*bitwidth))&mask;
    G[x] += qval*gap - max_abs_G;
  }
};


template <typename policy_t>
int ecq_r_body(
    uint8_t* in_uint8_t,
    int32_t in_uint8_t_size,
    float* out_float,
    int32_t out_float_size,
    int is_add_to,
    policy_t policy,
    void* stream) {

  uint8_t header[10];
  get_policy<policy_t>::memcpyOut(
    header,
    in_uint8_t,
    sizeof(uint8_t)*10,
    stream
  );
  uint8_t* Q = in_uint8_t + 10;
  float* G = out_float;
  int32_t in_size_used;
  float max_abs_G;
  memcpy(&in_size_used, header, sizeof(int32_t));
  memcpy(&max_abs_G, header+4, sizeof(float));
  uint8_t bitwidth = header[8];
  uint8_t unused_nums = header[9];
  int32_t M = in_size_used - 10;
  uint8_t data_per_byte = 8 / bitwidth;
  int32_t N = M * data_per_byte - unused_nums;
  uint8_t s = (1<<(bitwidth-1))-1;
  float gap = max_abs_G / s;
  if (unlikely(out_float_size < N)){
    printf("Output space too small: out_float_size < N.  %d vs. %d\n",
      static_cast<int32_t>(out_float_size),N);
    CHECK_EQ(0,1);
  }
  if (is_add_to){
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      generate_G_add_to(
        Q,
        G,
        data_per_byte,
        bitwidth,
        max_abs_G,
        gap
      )
    );
  }
  else{
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      generate_G(
        Q,
        G,
        data_per_byte,
        bitwidth,
        max_abs_G,
        gap
      )
    );
  }
  return 0;
}

} // namespace gradient_compression_body
} // namespace zq_cpp_lib

#endif
