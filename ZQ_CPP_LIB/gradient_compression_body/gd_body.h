#ifndef ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_GD_BODY_H
#define ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_GD_BODY_H
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
#include <vector>

namespace zq_cpp_lib {
namespace gradient_compression_body {

using namespace zq_cpp_lib::operate_memory;
struct generate_sample_G{
  float* sample_G;
  float* G;
  int32_t N;  //G_size
  uint64_t t;
  generate_sample_G(
    float* sample_G_,
    float* G_,
    int32_t N_,
    uint64_t t_
  ):
  sample_G(sample_G_),
  G(G_),
  N(N_),
  t(t_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    zq_cpp_lib::naive_int_random<uint32_t> r(0,N-1);
    r.srand(t+x);
    sample_G[x] = abs(G[r()]);
  }
};

struct generate_S_index{
  float* G;
  float threshold;
  generate_S_index(
    float* G_,
    float threshold_
  ):
  G(G_),
  threshold(threshold_)
  {}
  __host__ __device__
  bool operator()(const int32_t& x){
    return (G[x] > threshold) || (G[x] < -threshold);
  }
};
struct greater{
  const float threshold;
  greater(float t): threshold(t){}

  __host__ __device__
  bool operator()(const float&x) const {
    return (x>threshold) || (x<-threshold);
  }
};


struct cmp_float_data_by_int32_index{
  float* G;
  cmp_float_data_by_int32_index(float* G_)
  :G(G_){}
  __host__ __device__
  bool operator()(const int32_t&x, const int32_t& y){
    return abs(G[x]) > abs(G[y]);
  }
};

struct generate_S_value{
  int32_t *S_index;
  float* S_value;
  float* G;
  generate_S_value(
    int32_t* S_index_,
    float* S_value_,
    float* G_
  ):
  S_index(S_index_),
  S_value(S_value_),
  G(G_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    int32_t i = S_index[x];
    S_value[x] = G[i];
    G[i] = 0;
  }
};


template <typename policy_t>
int gd_body(
    float* in_float,
    int32_t N,
    float* residual,
    int32_t residual_size,
    uint8_t* out_uint8_t,
    double sample_rate,
    double drop_ratio,
    policy_t policy,
    void* stream) {
  double s_percent = 1 - drop_ratio;
  int32_t sample_cnt = static_cast<int32_t>(std::ceil(N*sample_rate));
  int32_t expected_selected = static_cast<int32_t>(std::ceil(N*s_percent));
  int32_t* header = reinterpret_cast<int32_t*>(out_uint8_t);
  float* sample_G = reinterpret_cast<float*>(out_uint8_t);
  float* G = in_float;
  float* R = residual;
  // printf("initialize over\n");
  thrust::transform(
    policy,
    G,
    G+N,
    R,
    R,
    thrust::plus<float>()
  );
  G=R;
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
        .count()
    )
  );
  // printf("generate_sample_G\n");

  thrust::sort(
    policy,
    sample_G,
    sample_G+sample_cnt,
    thrust::greater<float>()
  );
  // printf("sort sample_G\n");

  float threshold;
  int32_t threshold_index = static_cast<int32_t>(sample_cnt*s_percent);
  get_policy<policy_t>::memcpyOut(
    &threshold,
    sample_G + threshold_index,
    sizeof(float),
    stream
  );
  get_policy<policy_t>::streamSynchronize(stream);
  // printf("memcpyOut threshold\n");
  
  // get_policy<policy_t>::streamSynchronize(stream);
  int32_t* S_index = reinterpret_cast<int32_t*>(out_uint8_t+4);
  int32_t* S_index_end = thrust::copy_if(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    G,
    S_index,
    greater(threshold)
  );
  // printf("copy_if S_index\n");

  int32_t selected_num = S_index_end - S_index;
  // printf("selected_num=%d\texpected_selected=%d\tthreshold_index=%d\tthreshold=%f\n",selected_num,expected_selected,threshold_index,threshold);
  if (selected_num > expected_selected){
    thrust::sort(
      policy,
      S_index,
      S_index_end,
      cmp_float_data_by_int32_index(G)
    );
    selected_num = expected_selected;
    S_index_end = S_index + selected_num;
  }
  // printf("selected_num=%d\n",selected_num);
  // t_comment[index] = "sort S_index";
  // t_list[index++] = zq_cpp_lib::get_timestamp();
  // printf("sort S_index\n");
  
  int32_t out_size = 4 + selected_num*2*4;
  //printf("N=%d\tselected_num=%d\tout_size=%d\n", N, selected_num, out_size);
  get_policy<policy_t>::memcpyIn(
    header,
    &out_size,
    sizeof(int32_t),
    stream
  );
  get_policy<policy_t>::streamSynchronize(stream);
  // out_size = 0;
  // get_policy<policy_t>::memcpyOut(
  //   &out_size,
  //   header,
  //   sizeof(int32_t),
  //   stream
  // );
  // get_policy<policy_t>::streamSynchronize(stream);
  // if (unlikely(out_size!=2199)){
  //   printf("wrong compressed: out_size=%d\n",out_size);
  // }

  if (unlikely(selected_num == 0)){
    return 0;
  }
  // printf("memcpyIn header\n");

  float* S_value = reinterpret_cast<float*>(S_index_end);
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(selected_num),
    generate_S_value(
      S_index,
      S_value,
      G
    )
  );
  
  return 0;
}


struct generate_G_write_to{
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_write_to(
    float* G_,
    float* S_value_,
    int32_t* S_index_
  ):
  G(G_),
  S_value(S_value_),
  S_index(S_index_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    G[S_index[x]] = S_value[x];
  }
};
struct set_to_zero{
  float* a;
  set_to_zero(
    float* a_
  ){
    a=a_;
  };
  __host__ __device__
  void operator()(const int32_t&x){
    a[x] = 0;
  }
};
struct generate_G_add_to{
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_add_to(
    float* G_,
    float* S_value_,
    int32_t* S_index_
  ):
  G(G_),
  S_value(S_value_),
  S_index(S_index_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    G[S_index[x]] += S_value[x];
  }
};
template <typename policy_t>
int gd_r_body(
    std::vector<uint8_t*>& in_uint8_ts,
    std::vector<int32_t>& in_uint8_t_sizes,
    std::vector<float*>& out_floats,
    std::vector<int32_t>& out_float_sizes,
    std::vector<int>& is_add_tos,
    policy_t policy,
    void* stream){
 
  std::vector<int32_t> Ms;
  std::vector<int32_t*> headers;
  std::vector<float*> Gs;
  std::vector<int32_t*> S_indexes;
  std::vector<float*> S_values;
  std::vector<int32_t> S_index_sizes;

  
  auto _size = in_uint8_ts.size();
  for(int i = 0; i < _size; i++){
    int32_t M;
    Ms.push_back(M);
    headers.push_back(reinterpret_cast<int32_t*>(in_uint8_ts[i]));
    Gs.push_back(out_floats[i]);
    S_indexes.push_back(reinterpret_cast<int32_t*>(in_uint8_ts[i] + 4));


    get_policy<policy_t>::memcpyOut(&Ms[i], headers[i], sizeof(int32_t), stream);
  }

  get_policy<policy_t>::streamSynchronize(stream);


  for(int i = 0; i < _size; i++){
    S_index_sizes.push_back(((Ms[i] - 4) / 8));
    S_values.push_back(reinterpret_cast<float*>(S_indexes[i] + S_index_sizes[i]));
    if (unlikely(in_uint8_t_sizes[i] < Ms[i])) {
      printf("is_add_to : %d\n", is_add_tos[i]);
      printf(
        "input space provided is not enough! in_uint8_t_size=%d\tM=%d\n",
        static_cast<int32_t>(in_uint8_t_sizes[i]),
        Ms[i]);
      return -1;
    }
  }


  for(int i = 0; i < _size; i++){

    if (is_add_tos[i]) {
      thrust::for_each(
          policy,
          thrust::counting_iterator<int32_t>(0),
          thrust::counting_iterator<int32_t>(S_index_sizes[i]),
          generate_G_add_to(Gs[i], S_values[i], S_indexes[i]));
    } else {
      thrust::for_each(
        policy,
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(out_float_sizes[i]),
        set_to_zero(out_floats[i])
      );
      thrust::for_each(
          policy,
          thrust::counting_iterator<int32_t>(0),
          thrust::counting_iterator<int32_t>(S_index_sizes[i]),
          generate_G_write_to(Gs[i], S_values[i], S_indexes[i]));
    }
  }
  return 0;
}

} // namespace gradient_compression_body
} // namespace zq_cpp_lib

#endif
