#include "hp_cuda.h"

// gloabl status and thread manager
static GlobalStatus  global_status;
static ThreadManager tm1;
static ThreadManager tm2;
static ThreadManager tm3;
static ThreadManager tm4;
static ThreadManager tm5;
static FinishedQueues finished_queues_;

static TensorMap temp_cuda_tensors_for_c;



// CUDA  declarations
// terngrad
void hp_cuda_terngrad(
  std::vector<float*>& in_floats,
  std::vector<int32_t>& in_float_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<int32_t>& out_uint8_t_sizes,
  std::vector<uint8_t>& bitwidths,
  std::vector<int32_t>& enable_randoms,
  cudaStream_t stream
);

void hp_cuda_terngradr(
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
);

// tbq
void hp_cuda_tbqr(
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<float>& thresholds,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
);

void hp_cuda_tbq(
  std::vector<float*>& to_compress_floats,
  std::vector<int32_t>& to_compress_float_sizes,
  std::vector<float*>& residual_floats,
  std::vector<int32_t>& residual_float_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<int32_t>& out_uint8_t_sizes,
  std::vector<float>& thresholds,
  cudaStream_t stream
);

// grad drop
void hp_cuda_gd(
  std::vector<float*>& in_floats,
  std::vector<int32_t>& Ns,
  std::vector<float*>& residuals,
  std::vector<int32_t>& residual_sizes,
  std::vector<uint8_t*>& out_uint8_ts,
  std::vector<double>& sample_rates,
  std::vector<double>& drop_ratios,
  cudaStream_t stream
);

void hp_cuda_gdr(
  std::vector<uint8_t*>& in_uint8_ts,
  std::vector<int32_t>& in_uint8_t_sizes,
  std::vector<float*>& out_floats,
  std::vector<int32_t>& out_float_sizes,
  std::vector<int>& is_add_tos,
  cudaStream_t stream
);

// powersgd
void hp_cuda_powersgd_encode1(
  std::vector<std::shared_ptr<torch::Tensor>>& grads,
  std::vector<std::shared_ptr<torch::Tensor>>& residuals,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  c10::cuda::CUDAStream& stream
);
void hp_cuda_powersgd_encode2(
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  c10::cuda::CUDAStream& stream
);
void hp_cuda_powersgd_decode(
  std::vector<std::shared_ptr<torch::Tensor>>& Ps,
  std::vector<std::shared_ptr<torch::Tensor>>& Qs,
  std::vector<std::shared_ptr<torch::Tensor>>& Ms,
  std::vector<std::shared_ptr<torch::Tensor>>& residuals,
  std::vector<std::shared_ptr<torch::Tensor>>& grads,
  c10::cuda::CUDAStream& stream
);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_NOT_CUDA(x) TORCH_CHECK(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) CHECK_NOT_CUDA(x); CHECK_CONTIGUOUS(x)


// TaskQueue methods

bool TaskQueue::SubmitTask(Task t){
  std::lock_guard<std::mutex> guard(mutex_);
  task_queue_.push(t);
  return true;
}

std::vector<Task> TaskQueue::GetTasksFromQueue(){
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<Task> ret;
  while(!task_queue_.empty()){
    Task t = task_queue_.front();
    ret.push_back(std::move(t));
    task_queue_.pop();
  }
  //std::cout << "get " << ret.size() << " tasks from queue" << std::endl;
  return ret;
}

bool FinishQueue::PushFTKToFQ(std::string finished_task_key){
  std::lock_guard<std::mutex> guard(mutex_);
  finish_queue_.push(finished_task_key);
  return true;
}

std::vector<std::string> FinishQueue::GetFTKsFromFQ(){
  std::lock_guard<std::mutex> guard(mutex_);
  std::vector<std::string> ret;
  
  while(!finish_queue_.empty()){
    std::string s = finish_queue_.front();
    ret.push_back(std::move(s));
    finish_queue_.pop();
  }

  return ret;
}

std::shared_ptr<torch::Tensor> TensorMap::getTensor(std::string name, std::shared_ptr<torch::Tensor> target){
  std::lock_guard<std::mutex> guard(mutex_);
  if(tensor_map_.find(name) == tensor_map_.end()){
    tensor_map_[name] = std::make_shared<torch::Tensor>(target->cuda());
  }
  auto ret = tensor_map_[name];
  ret->copy_(*target, true);
  return ret;
}

bool RunLoopOnceForGradDrop(
  GlobalStatus& state, 
  ThreadManager& manager, 
  FinishedQueues &fq,
  c10::cuda::CUDAStream& data_stream,
  c10::cuda::CUDAStream& compute_stream
){
  if (manager.shut_down_) {
    return false;
  }
   
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = manager.thread_last_cycle_start_ +
                        std::chrono::microseconds(long(
                           500.)) -
                        start_time;
  //std::cout << "sleep time:" << std::endl;
  //std::cout << sleep_duration.count() << std::endl;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  manager.thread_last_cycle_start_ = std::chrono::steady_clock::now();
  
  
  auto task_list = manager.task_queue_.GetTasksFromQueue();
  //no task find
  if(task_list.size() == 0){
    return true;
  }


  std::vector<Task> comp_task;
  std::vector<Task> decomp_task;
  std::vector<Task> root_task;

  //graddrop compress
  std::vector<float*> in_floats;
  std::vector<int32_t> Ns;
  std::vector<float*> residuals;
  std::vector<int32_t> residual_sizes;
  std::vector<uint8_t*> out_uint8_ts;
  std::vector<double> sample_rates;
  std::vector<double> drop_ratios;

  //graddrop decompress
  std::vector<uint8_t*> in_uint8_ts;
  std::vector<int32_t> in_uint8_t_sizes;
  std::vector<float*> out_floats;
  std::vector<int32_t> out_float_sizes;
  std::vector<int> is_add_tos;

  for(auto t : task_list){
    switch (t.task_id){
      case COMP_TASK:{
        in_floats.push_back(t.tensor->data_ptr<float>());     
        Ns.push_back(t.tensor->numel());
        residuals.push_back(t.residual->data_ptr<float>());
        residual_sizes.push_back(t.residual->numel());
        out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        sample_rates.push_back(state.comp_alg_.sample_rate);
        drop_ratios.push_back(state.comp_alg_.drop_ratio);

        comp_task.push_back(t);
        break;
      }
      case DECOMP_TASK:{
        auto N = t.tensor->numel();
        int32_t compressed_size = 4 + int(2 * std::ceil(N * (1 - state.comp_alg_.drop_ratio))) * 4;
        t.B->slice(0,0,compressed_size,1).copy_(t.C->slice(0,0,compressed_size,1), true);
        in_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        in_uint8_t_sizes.push_back(compressed_size);
        out_floats.push_back(t.tensor->data_ptr<float>());
        out_float_sizes.push_back(t.tensor->numel());
        is_add_tos.push_back(t.is_root);

        decomp_task.push_back(t);
        break;
      } 
      case ROOT_DECOMP_COMP_TASK:{
        auto N = t.tensor->numel();
        int32_t compressed_size = 4 + int(2 * std::ceil(N * (1 - state.comp_alg_.drop_ratio))) * 4;
        auto _temp_tensor = temp_cuda_tensors_for_c.getTensor(t.tensor_name, t.C);


        for(int i = 0; i < state.size_; i++){
          if (i == state.rank_id_){
            continue;
          }
          auto head = i * compressed_size;
          auto tail = head + compressed_size;
          auto _temp = _temp_tensor->slice(0, head, tail, 1);
          in_uint8_ts.push_back(_temp.data_ptr<uint8_t>());
          in_uint8_t_sizes.push_back(_temp.numel());
          out_floats.push_back(t.tensor->data_ptr<float>());
          out_float_sizes.push_back(t.tensor->numel());
          is_add_tos.push_back(t.is_root);
        }

        in_floats.push_back(t.tensor->data_ptr<float>());     
        Ns.push_back(t.tensor->numel());
        residuals.push_back(t.residual->data_ptr<float>());
        residual_sizes.push_back(t.residual->numel());
        out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        sample_rates.push_back(state.comp_alg_.sample_rate);
        drop_ratios.push_back(state.comp_alg_.drop_ratio);

        root_task.push_back(t);
        break;
      }
    
      default:{
        std::cout << "error find task_id != 10 20 30 where task_id is " << t.task_id <<  std::endl;
        break;
      }
    }
  }


  //sync
  data_stream.synchronize();

  hp_cuda_gdr(
    in_uint8_ts,
    in_uint8_t_sizes,
    out_floats,
    out_float_sizes,
    is_add_tos,
    compute_stream
  );
  hp_cuda_gd(
    in_floats,
    Ns,
    residuals,
    residual_sizes,
    out_uint8_ts,
    sample_rates,
    drop_ratios,
    c10::cuda::getDefaultCUDAStream()
  );


  for(Task t : root_task){
    auto N = t.tensor->numel();
    int32_t compressed_size = 4 + int(2 * std::ceil(N * (1 - state.comp_alg_.drop_ratio))) * 4;
    t.C->slice(0, 0, compressed_size, 1).copy_(t.B->slice(0, 0, compressed_size, 1), true);
  }

  for(Task t : comp_task){
    auto N = t.tensor->numel();
    int32_t compressed_size = 4 + int(2 * std::ceil(N * (1 - state.comp_alg_.drop_ratio))) * 4;
    t.C->slice(0, 0, compressed_size, 1).copy_(t.B->slice(0, 0, compressed_size, 1), true);
  }

  data_stream.synchronize();

  for(Task t : decomp_task){
    fq.finished_queue_for_DECOMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : root_task){
    fq.finished_queue_for_ROOT_DECOMP_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : comp_task){
    fq.finished_queue_for_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }


  return true;
}


//backthread loop

bool RunLoopOnceForTerngrad(
  GlobalStatus& state, 
  ThreadManager& manager, 
  FinishedQueues &fq,
  c10::cuda::CUDAStream& data_stream,
  c10::cuda::CUDAStream& compute_stream
){

  if (manager.shut_down_) {
    return false;
  }
   
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = manager.thread_last_cycle_start_ +
                        std::chrono::microseconds(long(
                           500.)) -
                        start_time;
  //std::cout << "sleep time:" << std::endl;
  //std::cout << sleep_duration.count() << std::endl;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  manager.thread_last_cycle_start_ = std::chrono::steady_clock::now();
  
  
  auto task_list = manager.task_queue_.GetTasksFromQueue();
  //no task find
  if(task_list.size() == 0){
    return true;
  }
  
  std::vector<Task> comp_task;
  std::vector<Task> decomp_task;
  std::vector<Task> root_task;

  //compress
  std::vector<float*> in_floats;
  std::vector<int32_t> in_float_sizes;
  std::vector<uint8_t*> out_uint8_ts;
  std::vector<int32_t> out_uint8_t_sizes;
  std::vector<uint8_t> bitwidths;
  std::vector<int32_t> enable_randoms;

  //root compress
  std::vector<float*> root_in_floats;
  std::vector<int32_t> root_in_float_sizes;
  std::vector<uint8_t*> root_out_uint8_ts;
  std::vector<int32_t> root_out_uint8_t_sizes;
  std::vector<uint8_t> root_bitwidths;
  std::vector<int32_t> root_enable_randoms;


  //decompress
  std::vector<float*> out_floats;
  std::vector<int32_t> out_float_sizes;
  std::vector<uint8_t*> in_uint8_ts;
  std::vector<int32_t> in_uint8_t_sizes;
  std::vector<int> is_add_tos;


  for(auto t : task_list){
    switch (t.task_id){
      case COMP_TASK:{
        in_floats.push_back(t.tensor->data_ptr<float>());
        in_float_sizes.push_back(t.tensor->numel());
        out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        out_uint8_t_sizes.push_back(t.B->numel());
        bitwidths.push_back(state.comp_alg_.bitwidth);
        enable_randoms.push_back(state.comp_alg_.enable_random);
        comp_task.push_back(t);
        break;
      }
      case DECOMP_TASK:{
        auto compressed_size = t.B->numel();
        t.B->copy_(t.C->slice(0,0,compressed_size,1), true);

        out_floats.push_back(t.tensor->data_ptr<float>());
        out_float_sizes.push_back(t.tensor->numel());
        in_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        in_uint8_t_sizes.push_back(t.B->numel());
        is_add_tos.push_back(t.is_root);
        decomp_task.push_back(t);
        break;
      }
      case ROOT_DECOMP_COMP_TASK:{
        auto compressed_size = t.B->numel();
        auto _temp_tensor = temp_cuda_tensors_for_c.getTensor(t.tensor_name, t.C);
        
        for (int i = 0; i < state.size_; i++){
          if (i == state.rank_id_){
            continue;
          }
          auto head = i * compressed_size;
          auto tail = head + compressed_size;
          out_floats.push_back(t.tensor->data_ptr<float>());
          out_float_sizes.push_back(t.tensor->numel());

          auto _temp = _temp_tensor->slice(0, head, tail, 1);
          in_uint8_ts.push_back(_temp.data_ptr<uint8_t>());
          in_uint8_t_sizes.push_back(_temp.numel());
          is_add_tos.push_back(t.is_root);  
        }

        root_in_floats.push_back(t.tensor->data_ptr<float>());
        root_in_float_sizes.push_back(t.tensor->numel());
        root_out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        root_out_uint8_t_sizes.push_back(t.B->numel());
        root_bitwidths.push_back(state.comp_alg_.bitwidth);
        root_enable_randoms.push_back(state.comp_alg_.enable_random);
        
        root_task.push_back(t);
        break;
      }
      default:
        std::cout << "error find task_id != 10 20 30 where task_id is " << t.task_id <<  std::endl;
        break;
    }
  }

  //sync
  data_stream.synchronize();

  //begin task
  //decomp first


  //~root compress
  hp_cuda_terngrad(in_floats, in_float_sizes, out_uint8_ts, out_uint8_t_sizes, bitwidths, enable_randoms, compute_stream);

  for(Task t : comp_task){
    auto compressed_size = t.B->numel();
    t.C->slice(0, 0, compressed_size, 1).copy_(*(t.B));
    fq.finished_queue_for_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  } 


  //root and ~root decompress
  hp_cuda_terngradr(out_floats, out_float_sizes, in_uint8_ts, in_uint8_t_sizes, is_add_tos, compute_stream);

  //root compress
  hp_cuda_terngrad(root_in_floats, root_in_float_sizes, root_out_uint8_ts, root_out_uint8_t_sizes, root_bitwidths, root_enable_randoms, compute_stream);
  for(Task t : root_task){
    auto compressed_size = t.B->numel();
    t.C->slice(0, 0, compressed_size, 1).copy_(*(t.B));
    fq.finished_queue_for_ROOT_DECOMP_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  data_stream.synchronize();

  for(Task t : decomp_task){
    fq.finished_queue_for_DECOMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  return true;
}

bool RunLoopOnceForTBQ(
  GlobalStatus& state, 
  ThreadManager& manager, 
  FinishedQueues &fq,
  c10::cuda::CUDAStream& data_stream,
  c10::cuda::CUDAStream& compute_stream
){

  if (manager.shut_down_) {
    return false;
  }
   
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = manager.thread_last_cycle_start_ +
                        std::chrono::microseconds(long(
                           500.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  manager.thread_last_cycle_start_ = std::chrono::steady_clock::now();
  
  auto task_list = manager.task_queue_.GetTasksFromQueue();
  //no task find
  if(task_list.size() == 0){
    return true;
  }


  std::vector<Task> comp_task;
  std::vector<Task> decomp_task;
  std::vector<Task> root_task;

  //compress for tbq
  std::vector<float*> to_compress_floats;
  std::vector<int32_t> to_compress_float_sizes;
  std::vector<float*> residual_floats;
  std::vector<int32_t> residual_float_sizes;
  std::vector<uint8_t*> out_uint8_ts;
  std::vector<int32_t> out_uint8_t_sizes;
  std::vector<float> thresholds_comp;
  

  //decompress for tbq
  std::vector<float*> out_floats;
  std::vector<int32_t> out_float_sizes;
  std::vector<uint8_t*> in_uint8_ts;
  std::vector<int32_t> in_uint8_t_sizes;
  std::vector<float> thresholds_decomp;
  std::vector<int> is_add_tos;


  for(auto t : task_list){
    switch (t.task_id){
      case COMP_TASK:{
        to_compress_floats.push_back(t.tensor->data_ptr<float>());
        to_compress_float_sizes.push_back(t.tensor->numel());
        residual_floats.push_back(t.residual->data_ptr<float>());
        residual_float_sizes.push_back(t.residual->numel());
        out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        out_uint8_t_sizes.push_back(t.B->numel());
        thresholds_comp.push_back(state.comp_alg_.threshold);
        comp_task.push_back(t);
        break;
      }
      case DECOMP_TASK:{
        auto compressed_size = t.B->numel();
        t.B->copy_(t.C->slice(0,0,compressed_size,1), true);

        out_floats.push_back(t.tensor->data_ptr<float>());
        out_float_sizes.push_back(t.tensor->numel());
        in_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        in_uint8_t_sizes.push_back(t.B->numel());
        thresholds_decomp.push_back(state.comp_alg_.threshold);
        is_add_tos.push_back(t.is_root);
        decomp_task.push_back(t);
        break;
      }
      case ROOT_DECOMP_COMP_TASK:{
        auto compressed_size = t.B->numel();
        auto _temp_tensor = temp_cuda_tensors_for_c.getTensor(t.tensor_name, t.C);
        
        for (int i = 0; i < state.size_; i++){
          if (i == state.rank_id_){
            continue;
          }
          auto head = i * compressed_size;
          auto tail = head + compressed_size;
          out_floats.push_back(t.tensor->data_ptr<float>());
          out_float_sizes.push_back(t.tensor->numel());

          auto _temp = _temp_tensor->slice(0, head, tail, 1);
          in_uint8_ts.push_back(_temp.data_ptr<uint8_t>());
          in_uint8_t_sizes.push_back(_temp.numel());
          thresholds_decomp.push_back(state.comp_alg_.threshold);
          is_add_tos.push_back(t.is_root);  
        }

        to_compress_floats.push_back(t.tensor->data_ptr<float>());
        to_compress_float_sizes.push_back(t.tensor->numel());
        residual_floats.push_back(t.residual->data_ptr<float>());
        residual_float_sizes.push_back(t.residual->numel());
        out_uint8_ts.push_back(t.B->data_ptr<uint8_t>());
        out_uint8_t_sizes.push_back(t.B->numel());
        thresholds_comp.push_back(state.comp_alg_.threshold);

        root_task.push_back(t);
        break;
      }
      default:
        std::cout << "error find task_id != 10 20 30 where task_id is " << t.task_id <<  std::endl;
        break;
    }
  }

  data_stream.synchronize();

  hp_cuda_tbqr(out_floats, out_float_sizes, in_uint8_ts, in_uint8_t_sizes, thresholds_decomp, is_add_tos, compute_stream);
  hp_cuda_tbq(to_compress_floats, to_compress_float_sizes, residual_floats, residual_float_sizes, out_uint8_ts, out_uint8_t_sizes, thresholds_comp, compute_stream);


  for(Task t : root_task){
    auto compressed_size = t.B->numel();
    t.C->slice(0, 0, compressed_size, 1).copy_(*(t.B), true);
  }

  for(Task t : comp_task){
    auto compressed_size = t.B->numel();
    t.C->slice(0, 0, compressed_size, 1).copy_(*(t.B), true);
  }

  data_stream.synchronize();

  for(Task t : decomp_task){
    fq.finished_queue_for_DECOMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : root_task){
    fq.finished_queue_for_ROOT_DECOMP_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : comp_task){
    fq.finished_queue_for_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }
  return true;
}

bool RunLoopOnceForPowerSGD(
  GlobalStatus& state, 
  ThreadManager& manager, 
  FinishedQueues &fq,
  c10::cuda::CUDAStream& data_stream,
  c10::cuda::CUDAStream& compute_stream
){
  if (manager.shut_down_) {
    return false;
  }
   
  auto start_time = std::chrono::steady_clock::now();
  auto sleep_duration = manager.thread_last_cycle_start_ +
                        std::chrono::microseconds(long(
                           500.)) -
                        start_time;
  if (sleep_duration > std::chrono::steady_clock::duration::zero()) {
    std::this_thread::sleep_for(sleep_duration);
  }
  manager.thread_last_cycle_start_ = std::chrono::steady_clock::now();
  
  auto task_list = manager.task_queue_.GetTasksFromQueue();
  //no task find
  if(task_list.size() == 0){
    return true;
  }

  std::vector<Task> encode1_task;
  std::vector<Task> encode2_task;
  std::vector<Task> decode_task;

  // encode1 params
  std::vector<std::shared_ptr<torch::Tensor>> encode1_input_grads;
  std::vector<std::shared_ptr<torch::Tensor>> encode1_input_residuals;
  std::vector<std::shared_ptr<torch::Tensor>> encode1_input_Qs;
  std::vector<std::shared_ptr<torch::Tensor>> encode1_output_Ms;
  std::vector<std::shared_ptr<torch::Tensor>> encode1_output_Ps;

  // encode2 params
  std::vector<std::shared_ptr<torch::Tensor>> encode2_input_Ps;
  std::vector<std::shared_ptr<torch::Tensor>> encode2_input_Ms;
  std::vector<std::shared_ptr<torch::Tensor>> encode2_output_Qs;

  // decode params
  std::vector<std::shared_ptr<torch::Tensor>> decode_input_Ps;
  std::vector<std::shared_ptr<torch::Tensor>> decode_input_Qs;
  std::vector<std::shared_ptr<torch::Tensor>> decode_output_Ms;
  std::vector<std::shared_ptr<torch::Tensor>> decode_output_residuals;
  std::vector<std::shared_ptr<torch::Tensor>> decode_output_grads;

  for (auto t: task_list) {
    switch (t.task_id)
    {
    case POWERSGD_ENCODE1:
      // std::cout << "received encode1 task! " << t.tensor_name << std::endl;
      encode1_input_grads.push_back(t.tensor);
      encode1_input_residuals.push_back(t.residual);
      encode1_input_Qs.push_back(t.Q);
      encode1_output_Ms.push_back(t.M);
      encode1_output_Ps.push_back(t.P);
      encode1_task.push_back(t);
      break;

    case POWERSGD_ENCODE2:
      // std::cout << "received encode2 task! " << t.tensor_name << std::endl;
      encode2_input_Ps.push_back(t.P);
      encode2_input_Ms.push_back(t.M);
      encode2_output_Qs.push_back(t.Q);
      encode2_task.push_back(t);
      break;

    case POWERSGD_DECODE:
      // std::cout << "received decode task! " << t.tensor_name << std::endl;
      decode_input_Ps.push_back(t.P);
      decode_input_Qs.push_back(t.Q);
      decode_output_Ms.push_back(t.M);
      decode_output_residuals.push_back(t.residual);
      decode_output_grads.push_back(t.tensor);
      decode_task.push_back(t);
      break;

    default:
      std::cout << "error find task_id != 40 50 60 where task_id is " << t.task_id <<  std::endl;
      break;
    }
  }

  data_stream.synchronize();

  hp_cuda_powersgd_encode1(
    encode1_input_grads,
    encode1_input_residuals,
    encode1_input_Qs,
    encode1_output_Ms,
    encode1_output_Ps,
    compute_stream
  );

  hp_cuda_powersgd_encode2(
    encode2_input_Ps,
    encode2_input_Ms,
    encode2_output_Qs,
    compute_stream
  );

  hp_cuda_powersgd_decode(
    decode_input_Ps,
    decode_input_Qs,
    decode_output_Ms,
    decode_output_residuals,
    decode_output_grads,
    compute_stream
  );

  data_stream.synchronize();

  for(Task t : decode_task){
    // std::cout << "finished decode " << t.tensor_name << std::endl;
    fq.finished_queue_for_DECOMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : encode1_task){
    // std::cout << "finished encode1 " << t.tensor_name << std::endl;
    fq.finished_queue_for_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  for(Task t : encode2_task){
    // std::cout << "finished encode2 " << t.tensor_name << std::endl;
    fq.finished_queue_for_ROOT_DECOMP_COMP_TASK_.PushFTKToFQ(t.tensor_name);
  }

  return true;
}


//task_id = 1 for comp; task_id = 2 for decomp;
void BackgroundThreadLoop(GlobalStatus& state, ThreadManager& manager, FinishedQueues &fq, int32_t device_id){
  std::cout << device_id << std::endl;
  auto device = c10::Device(c10::DeviceType::CUDA, device_id);
  c10::cuda::OptionalCUDAGuard d;
  d.reset_device(device);
  
  auto stream = c10::cuda::getStreamFromPool(true, device.index());
  
  c10::cuda::OptionalCUDAStreamGuard s;
  s.reset_stream(c10::cuda::CUDAStream(stream));
  
  std::function<bool(GlobalStatus&, ThreadManager&, FinishedQueues&, c10::cuda::CUDAStream&, c10::cuda::CUDAStream&)> loop_function;
  c10::cuda::CUDAStream data_stream_ = stream;
  static c10::cuda::CUDAStream compute_stream_ = c10::cuda::getStreamFromPool(true, device.index());
  
  

  if(state.comp_alg_.CompAlgName == "terngrad"){
    loop_function = RunLoopOnceForTerngrad;
  } else if (state.comp_alg_.CompAlgName == "graddrop") {
    loop_function = RunLoopOnceForGradDrop;
  } else if (state.comp_alg_.CompAlgName == "powersgd") {
    loop_function = RunLoopOnceForPowerSGD;
  } else {
    loop_function = RunLoopOnceForTBQ;
  }

  manager.initialize_done_ = true;

  try {
    while (loop_function(std::ref(state), std::ref(manager), std::ref(fq), std::ref(data_stream_), std::ref(compute_stream_)) && !manager.shut_down_);
  } catch (const std::exception& ex) {
    std::cout << "background loop for comp uncaught exception: " << ex.what() << std::endl;
  }
}


bool start_thread_(ThreadManager& tm, int device_id){
  if (!tm.initialize_done_){
      tm.thread_ = std::thread(BackgroundThreadLoop, 
      std::ref(global_status), 
      std::ref(tm),
      std::ref(finished_queues_),
      device_id);
  }

  while(!tm.initialize_done_){
    std::cout << "init backthread not finished" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return true;
}

bool init(
  std::string alg_name_, 
  std::unordered_map<std::string, double> alg_paras, 
  int32_t device_id, 
  int32_t rank_id, 
  int32_t size
){
  if (alg_name_ == "terngrad"){
    global_status.comp_alg_.CompAlgName = alg_name_;
    global_status.comp_alg_.enable_random = alg_paras["enable_random"];
    global_status.comp_alg_.bitwidth = alg_paras["bitwidth"];
  } else if (alg_name_ == "tbq"){
    global_status.comp_alg_.CompAlgName = alg_name_;
    global_status.comp_alg_.threshold = alg_paras["threshold"];
    global_status.multithread_ = false;
  } else if (alg_name_ == "graddrop"){
    global_status.comp_alg_.CompAlgName = alg_name_;
    global_status.comp_alg_.drop_ratio = alg_paras["drop_ratio"];
    global_status.comp_alg_.sample_rate = alg_paras["sample_rate"];
  } else if (alg_name_ == "powersgd"){
    global_status.comp_alg_.CompAlgName = alg_name_;
    global_status.comp_alg_.matrix_approximation_rank = alg_paras["matrix_approximation_rank"];
  } else {
    std::cout << "only support terngrad, tbq, graddrop and powersgd now" << std::endl;
    return false;
  }

  if(global_status.multithread_){
    std::cout << "use multithread for compression, thread num : " << BACK_THREAD_ << std::endl;
  }
  

  global_status.device_id = device_id;
  global_status.rank_id_ = rank_id;
  global_status.size_ = size;
  global_status.round_robin = 0;

  
  start_thread_(tm1, device_id);
  if(global_status.multithread_){
    start_thread_(tm2, device_id);
    start_thread_(tm3, device_id);
    start_thread_(tm4, device_id);
    start_thread_(tm5, device_id);
  }
 
  return true;
}

void shut_down_thread_(ThreadManager& tm){
  tm.shut_down_ = true;
  if(tm.thread_.joinable()){
    tm.thread_.join();
  }
}

bool end(){
  shut_down_thread_(tm1);
  shut_down_thread_(tm2);
  shut_down_thread_(tm3);
  shut_down_thread_(tm4);
  shut_down_thread_(tm5);
  return true;
}

void inline check_thread(){
  CHECK(tm1.initialize_done_ && !tm1.shut_down_);
  if (global_status.multithread_){
    CHECK(tm2.initialize_done_ && !tm2.shut_down_);
    CHECK(tm3.initialize_done_ && !tm3.shut_down_);
    CHECK(tm4.initialize_done_ && !tm4.shut_down_);
    CHECK(tm5.initialize_done_ && !tm5.shut_down_);
  }
}

void inline submit_task(Task t){
  if (!global_status.multithread_){
    tm1.task_queue_.SubmitTask(t);
    return;
  }
  switch (global_status.round_robin)
  {
  case 0:
    tm1.task_queue_.SubmitTask(t);
    break;
  case 1:
    tm2.task_queue_.SubmitTask(t);
    break;
  case 2:
    tm3.task_queue_.SubmitTask(t);
    break;
  case 3:
    tm4.task_queue_.SubmitTask(t);
    break;
  case 4:
    tm5.task_queue_.SubmitTask(t);
    break;
  default:
    std::cout << "error in submit task" << std::endl;
    break;
  }
  global_status.round_robin = (global_status.round_robin + 1) % BACK_THREAD_;
}

bool submit_task_with_residual(
  std::string tensor_name,
  torch::Tensor tensor,
  torch::Tensor residual,
  torch::Tensor B,
  torch::Tensor C,
  int is_root,
  int task_id
){
  CHECK_CUDA_INPUT(tensor);
  CHECK_CUDA_INPUT(residual);
  CHECK_CUDA_INPUT(B);
  CHECK_CPU_INPUT(C);

  if(is_root != 0 && is_root != 1){
    std::cout << "is_root should be 0 (~root) or 1 (root), find is_root is " << is_root << std::endl;
    return false;
  }
  if(task_id != COMP_TASK && task_id != DECOMP_TASK && task_id != ROOT_DECOMP_COMP_TASK){
    std::cout << "'submit_task' : task_id should be 10 (comp) or 20 (decomp) or 30 (root decomp and comp), find task_id is " << task_id << std::endl;
    return false;
  }

  Task t;
  t.tensor_name = tensor_name;
  t.tensor = std::make_shared<torch::Tensor>(tensor);
  t.residual = std::make_shared<torch::Tensor>(residual);
  t.B = std::make_shared<torch::Tensor>(B);
  t.C = std::make_shared<torch::Tensor>(C);
  t.is_root = is_root;
  t.task_id = task_id;

  check_thread(); 
  submit_task(t);
  return true;
}


bool submit_task_without_residual(
  std::string tensor_name,
  torch::Tensor tensor,
  torch::Tensor B,
  torch::Tensor C,
  int is_root,
  int task_id
){
  CHECK_CUDA_INPUT(tensor);
  CHECK_CUDA_INPUT(B);
  CHECK_CPU_INPUT(C);

  if(is_root != 0 && is_root != 1){
    std::cout << "is_root should be 0 (~root) or 1 (root), find is_root is " << is_root << std::endl;
    return false;
  }
  if(task_id != COMP_TASK && task_id != DECOMP_TASK && task_id != ROOT_DECOMP_COMP_TASK){
    std::cout << "'submit_task' : task_id should be 10 (comp) or 20 (decomp) or 30 (root decomp and comp), find task_id is " << task_id << std::endl;
    return false;
  }

  Task t;
  t.tensor_name = tensor_name;
  t.tensor = std::make_shared<torch::Tensor>(tensor);
  t.residual = nullptr;
  t.B = std::make_shared<torch::Tensor>(B);
  t.C = std::make_shared<torch::Tensor>(C);
  t.is_root = is_root;
  t.task_id = task_id;

  check_thread();  
  submit_task(t);
  return true;
}

bool submit_task_for_powersgd(
  std::string tensor_name,
  torch::Tensor tensor,
  torch::Tensor residual,
  torch::Tensor M,
  torch::Tensor P,
  torch::Tensor Q,
  int task_id
) {
  CHECK_CUDA_INPUT(tensor);
  CHECK_CUDA_INPUT(residual);
  CHECK_CUDA_INPUT(M);
  CHECK_CUDA_INPUT(P);
  CHECK_CUDA_INPUT(Q);

  if(task_id != POWERSGD_ENCODE1 && task_id != POWERSGD_ENCODE2 && task_id != POWERSGD_DECODE){
    std::cout << "'submit_task' : powersgd task_id should be 40 (encode1) or 50 (encode2) or 60 (decode), find task_id is " << task_id << std::endl;
    return false;
  }

  Task t;
  t.tensor_name = tensor_name;
  t.tensor = std::make_shared<torch::Tensor>(tensor);
  t.residual = std::make_shared<torch::Tensor>(residual);
  t.M = std::make_shared<torch::Tensor>(M);
  t.P = std::make_shared<torch::Tensor>(P);
  t.Q = std::make_shared<torch::Tensor>(Q);
  t.task_id = task_id;

  check_thread(); 
  submit_task(t);
  return true;
}

std::vector<std::string> getResults(int task_id){
  std::vector<std::string> result;
  switch (task_id)
  {
  case COMP_TASK: case POWERSGD_ENCODE1:
    result = finished_queues_.finished_queue_for_COMP_TASK_.GetFTKsFromFQ();
    break;
  
  case DECOMP_TASK: case POWERSGD_DECODE:
    result = finished_queues_.finished_queue_for_DECOMP_TASK_.GetFTKsFromFQ();
    break;

  case ROOT_DECOMP_COMP_TASK: case POWERSGD_ENCODE2:
    result = finished_queues_.finished_queue_for_ROOT_DECOMP_COMP_TASK_.GetFTKsFromFQ();
    break;

  default:
    std::cout << "not invalid task_id in getResults: " << task_id << std::endl;
  }
  return result;
}












PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init",  &init, "init");
  m.def("end", &end, "end");
  m.def("getResults",  &getResults, "getResults");
  m.def("submit_task",  &submit_task_without_residual, "submit_task without residual");
  m.def("submit_task",  &submit_task_with_residual, "submit_task with residual");
  m.def("submit_task",  &submit_task_for_powersgd, "submit_task for powersgd");
}