#ifndef ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_DGC_BODY_H
#define ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_DGC_BODY_H
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
struct generate_sample_G {
  float* sample_G;
  float* G;
  int32_t N; // G_size
  uint64_t t;
  generate_sample_G(float* sample_G_, float* G_, int32_t N_, uint64_t t_)
      : sample_G(sample_G_), G(G_), N(N_), t(t_) {}
  __host__ __device__ void operator()(const int32_t& x) {
    zq_cpp_lib::naive_int_random<uint32_t> r(0, N - 1);
    r.srand(t + x);
    sample_G[x] = abs(G[r()]);
  }
};

struct generate_S_index {
  float* G;
  float threshold;
  generate_S_index(float* G_, float threshold_)
      : G(G_), threshold(threshold_) {}
  __host__ __device__ bool operator()(const int32_t& x) {
    return (G[x] > threshold) || (G[x] < -threshold);
  }
};
struct greater {
  const float threshold;
  greater(float t) : threshold(t) {}

  __host__ __device__ bool operator()(const float& x) const {
    return (x > threshold) || (x < -threshold);
  }
};

struct cmp_float_data_by_int32_index {
  float* G;
  cmp_float_data_by_int32_index(float* G_) : G(G_) {}
  __host__ __device__ bool operator()(const int32_t& x, const int32_t& y) {
    return abs(G[x]) > abs(G[y]);
  }
};

struct generate_S_value {
  int32_t* S_index;
  float* S_value;
  float* G;
  generate_S_value(int32_t* S_index_, float* S_value_, float* G_)
      : S_index(S_index_), S_value(S_value_), G(G_) {}
  __host__ __device__ void operator()(const int32_t& x) {
    int32_t i = S_index[x];
    S_value[x] = G[i];
    G[i] = 0;
  }
};

template <typename policy_t>
int dgc_with_auxiliary_body(
    float* in_float,
    int N,
    uint8_t* auxiliary_uint8_t,
    uint8_t* out_uint8_t,
    double sample_rate,
    double s_percent,
    policy_t policy,
    void* stream) {
  int32_t sample_cnt = static_cast<int32_t>(std::ceil(N * sample_rate));
  int32_t expected_selected = static_cast<int32_t>(std::ceil(N * s_percent));
  int32_t* header = reinterpret_cast<int32_t*>(out_uint8_t);
  float* sample_G = reinterpret_cast<float*>(auxiliary_uint8_t);
  float* G = in_float;
  thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(sample_cnt),
      generate_sample_G(
          sample_G,
          G,
          N,
          std::chrono::high_resolution_clock::now()
              .time_since_epoch()
              .count()));
  // zt.record("generate_sample_G");

  thrust::sort(
      policy, sample_G, sample_G + sample_cnt, thrust::greater<float>());
  // zt.record("sort sample_G");

  float threshold;
  int32_t threshold_index = static_cast<int32_t>(sample_cnt * s_percent);
  get_policy<policy_t>::memcpyOut(
      &threshold, sample_G + threshold_index, sizeof(float), stream);
  // zt.record("memcpyOut threshold");

  get_policy<policy_t>::streamSynchronize(stream);
  int32_t* S_index = reinterpret_cast<int32_t*>(auxiliary_uint8_t);
  int32_t* S_index_end = thrust::copy_if(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      G,
      S_index,
      greater(threshold));
  // zt.record("copy_if S_index");

  int32_t selected_num = S_index_end - S_index;
  // printf("selected_num=%d\texpected_selected=%d\tthreshold_index=%d\tthreshold=%f\n",selected_num,expected_selected,threshold_index,threshold);
  if (selected_num > expected_selected) {
    thrust::sort(
        policy, S_index, S_index_end, cmp_float_data_by_int32_index(G));
    selected_num = expected_selected;
    S_index_end = S_index + selected_num;
  }
  get_policy<policy_t>::memcpyDoubleIn(
    out_uint8_t+4, S_index, sizeof(int32_t)*selected_num, stream
  );
  S_index = reinterpret_cast<int32_t*>(out_uint8_t+4);
  S_index_end = S_index + selected_num;
  // printf("selected_num=%d\n",selected_num);
  // zt.record("sort S_index");

  int32_t out_size = 4 + selected_num * 2 * 4;
  // printf("%s[LINE:%d]: out_size=%d\tselected_num=%d\texpected_selected=%d\tinput_size=%d\n", __FILE__, __LINE__, out_size,
  //   selected_num, expected_selected, N);
  get_policy<policy_t>::memcpyIn(header, &out_size, sizeof(int32_t), stream);
  if (unlikely(selected_num == 0)) {
    return 0;
  }
  // zt.record("memcpyIn header");

  float* S_value = reinterpret_cast<float*>(S_index_end);
  thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(selected_num),
      generate_S_value(S_index, S_value, G));
  return 0;
}

template <typename policy_t>
int dgc_body(
    float* in_float,
    int N,
    uint8_t* out_uint8_t,
    double sample_rate,
    double s_percent,
    policy_t policy,
    void* stream) {
  int32_t sample_cnt = static_cast<int32_t>(std::ceil(N * sample_rate));
  int32_t expected_selected = static_cast<int32_t>(std::ceil(N * s_percent));
  int32_t* header = reinterpret_cast<int32_t*>(out_uint8_t);
  float* sample_G = reinterpret_cast<float*>(out_uint8_t);
  float* G = in_float;
  thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(sample_cnt),
      generate_sample_G(
          sample_G,
          G,
          N,
          std::chrono::high_resolution_clock::now()
              .time_since_epoch()
              .count()));
  // zt.record("generate_sample_G");

  thrust::sort(
      policy, sample_G, sample_G + sample_cnt, thrust::greater<float>());
  // zt.record("sort sample_G");

  float threshold;
  int32_t threshold_index = static_cast<int32_t>(sample_cnt * s_percent);
  get_policy<policy_t>::memcpyOut(
      &threshold, sample_G + threshold_index, sizeof(float), stream);
  // zt.record("memcpyOut threshold");

  get_policy<policy_t>::streamSynchronize(stream);
  int32_t* S_index = reinterpret_cast<int32_t*>(out_uint8_t + 4);
  int32_t* S_index_end = thrust::copy_if(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      G,
      S_index,
      greater(threshold));
  // zt.record("copy_if S_index");

  int32_t selected_num = S_index_end - S_index;
  // printf("selected_num=%d\texpected_selected=%d\tthreshold_index=%d\tthreshold=%f\n",selected_num,expected_selected,threshold_index,threshold);
  if (selected_num > expected_selected) {
    thrust::sort(
        policy, S_index, S_index_end, cmp_float_data_by_int32_index(G));
    selected_num = expected_selected;
    S_index_end = S_index + selected_num;
  }
  // printf("selected_num=%d\n",selected_num);
  // zt.record("sort S_index");

  int32_t out_size = 4 + selected_num * 2 * 4;
  // printf("%s[LINE:%d]: out_size=%d\tselected_num=%d\texpected_selected=%d\tinput_size=%d\n", __FILE__, __LINE__, out_size,
  //   selected_num, expected_selected, N);
  get_policy<policy_t>::memcpyIn(header, &out_size, sizeof(int32_t), stream);
  if (unlikely(selected_num == 0)) {
    return 0;
  }
  // zt.record("memcpyIn header");

  float* S_value = reinterpret_cast<float*>(S_index_end);
  thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(selected_num),
      generate_S_value(S_index, S_value, G));
  return 0;
}

struct generate_G_write_to {
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_write_to(float* G_, float* S_value_, int32_t* S_index_)
      : G(G_), S_value(S_value_), S_index(S_index_) {}
  __host__ __device__ void operator()(const int32_t& x) {
    G[S_index[x]] = S_value[x];
  }
};

struct generate_G_add_to {
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_add_to(float* G_, float* S_value_, int32_t* S_index_)
      : G(G_), S_value(S_value_), S_index(S_index_) {}
  __host__ __device__ void operator()(const int32_t& x) {
    G[S_index[x]] += S_value[x];
  }
};

template <typename policy_t>
int dgc_r_body(
    uint8_t* in_uint8_t,
    int32_t in_uint8_t_size,
    float* out_float,
    int32_t out_float_size,
    int is_add_to,
    policy_t policy,
    void* stream) {
  int32_t M;
  int32_t* header = reinterpret_cast<int32_t*>(in_uint8_t);
  float* G = out_float;
  int32_t* S_index = reinterpret_cast<int32_t*>(in_uint8_t + 4);

  get_policy<policy_t>::memcpyOut(&M, header, sizeof(int32_t), stream);
  int32_t S_index_size = (M - 4) / 8;
  float* S_value = reinterpret_cast<float*>(S_index + S_index_size);
  if (unlikely(in_uint8_t_size < M)) {
    printf(
        "input space provided is not enough! in_uint8_t_size=%d\tM=%d\n",
        static_cast<int32_t>(in_uint8_t_size),
        M);
    return -1;
  }
  // we don't check outputspace is enough or not, user should be careful about
  // this.
  if (is_add_to) {
    thrust::for_each(
        policy,
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(S_index_size),
        generate_G_add_to(G, S_value, S_index));
  } else {
    thrust::for_each(
        policy,
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(S_index_size),
        generate_G_write_to(G, S_value, S_index));
  }
  return 0;
}

} // namespace gradient_compression_body
} // namespace zq_cpp_lib

#endif
