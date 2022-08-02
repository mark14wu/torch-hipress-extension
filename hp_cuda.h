#ifndef hp_CUDA_H
#define hp_CUDA_H

#include <torch/extension.h>
#include <vector>
#include <thread>
#include <atomic>
#include <queue>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <assert.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <c10/core/Device.h> 
#include <c10/core/DefaultDtype.h>
#include <c10/core/DeviceGuard.h>
#include <c10/cuda/CUDAGuard.h>

#define COMP_TASK 10
#define DECOMP_TASK 20
#define ROOT_DECOMP_COMP_TASK 30
#define POWERSGD_ENCODE1 40
#define POWERSGD_ENCODE2 50
#define POWERSGD_DECODE 60
#define BACK_THREAD_ 5
struct Task
{
    std::string tensor_name; //tab
    //grad
    std::shared_ptr<torch::Tensor> tensor;
    //residual (optional)
    std::shared_ptr<torch::Tensor> residual;
    //on gpu
    std::shared_ptr<torch::Tensor> B;
    //on cpu
    std::shared_ptr<torch::Tensor> C;
    //for powersgd
    std::shared_ptr<torch::Tensor> M;
    std::shared_ptr<torch::Tensor> P;
    std::shared_ptr<torch::Tensor> Q;
    //is root
    // 1 is root
    // 0 is ~root
    int is_root;
     //TASK_id
    // 10 for comp; 20 for decomp
    int task_id;
};


struct CompAlg {
  std::string CompAlgName;
  //for terngrad
  int enable_random;
  int bitwidth;
  //for tbq
  float threshold;
  //for graddrop
  double sample_rate;
  double drop_ratio;
  //for powersgd
  int matrix_approximation_rank;
};

class TaskQueue {
public:
    TaskQueue() = default;
    //TaskQueue(const TaskQueue&) = delete;
    bool SubmitTask(Task t);

    std::vector<Task> GetTasksFromQueue();
    
protected:

  // task_queue to submit and get tasks
  std::queue<Task> task_queue_;

  // A mutex that needs to be used whenever operations on message queue are
  // done.
  mutable std::mutex mutex_;
};

class FinishQueue{
public:
  FinishQueue() = default;
  FinishQueue(const FinishQueue&) = delete;

  bool PushFTKToFQ(std::string finished_task_key);
  std::vector<std::string> GetFTKsFromFQ();

protected:
  std::queue<std::string> finish_queue_;
  mutable std::mutex mutex_;
};

class TensorMap{
public:
  TensorMap() = default;
  TensorMap(const TensorMap&) = delete;

  std::shared_ptr<torch::Tensor> getTensor(std::string name, std::shared_ptr<torch::Tensor> target);

protected:
  std::unordered_map<std::string, std::shared_ptr<torch::Tensor>> tensor_map_;
  mutable std::mutex mutex_;
  std::atomic_bool flag_{false};
};

struct ThreadManager{
  std::thread thread_;
  std::atomic_bool initialize_done_{false};
  TaskQueue task_queue_;
  std::atomic_bool shut_down_{false};
  std::chrono::steady_clock::time_point thread_last_cycle_start_;

  ~ThreadManager() {
    shut_down_ = true;
    if (thread_.joinable()){
        thread_.join();
      }
  }
};


struct FinishedQueues{
  FinishQueue finished_queue_for_COMP_TASK_;
  FinishQueue finished_queue_for_DECOMP_TASK_;
  FinishQueue finished_queue_for_ROOT_DECOMP_COMP_TASK_;
};



struct GlobalStatus
{
    //record messages for compression alg
    CompAlg comp_alg_;
    int32_t device_id;
    int32_t size_;
    int32_t rank_id_;
    int32_t round_robin{0};
    bool multithread_{true};
};



#endif