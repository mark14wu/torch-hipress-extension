#ifndef _TIME_COST_HPP_
#define _TIME_COST_HPP_
#include <sys/time.h>
#include <unistd.h>
#include <string>
#include <vector>


namespace zq_cpp_lib{
  inline struct timeval get_timestamp(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t;
  }
  inline double get_cost_time_by_s(struct timeval t1, struct timeval t2){
    double ans = (t2.tv_sec-t1.tv_sec)*1.0+(t2.tv_usec-t1.tv_usec)/1000000.0;
    return ans;
  }
  inline double get_cost_time_by_ms(struct timeval t1, struct timeval t2){
    double ans = (t2.tv_sec-t1.tv_sec)*1000.0+(t2.tv_usec-t1.tv_usec)/1000.0;
    return ans;
  }
  inline double get_cost_time_by_us(struct timeval t1, struct timeval t2){
    double ans = (t2.tv_sec-t1.tv_sec)*1000000.0+(t2.tv_usec-t1.tv_usec)/1.0;
    return ans;
  }
  struct time_cost{
    private:
      std::vector<struct timeval>t_list;
      std::vector<std::string>t_comment;
      struct timeval t;
    public:
      inline void start(){
        gettimeofday(&t,NULL);
        t_list.push_back(t);
      }
      inline void record(std::string comment=""){
        gettimeofday(&t,NULL);
        t_list.push_back(t);
        t_comment.push_back(comment);
      }
      inline void print_by_us(){
        for (uint32_t i = 0; i < t_comment.size(); i++){
          printf("[%s]%.0lf\t",t_comment[i].c_str(),get_cost_time_by_us(t_list[i],t_list[i+1]));
        }
        printf("[total]%.0lf\n", get_cost_time_by_us(t_list[0],t_list[t_comment.size()]));
      }
      inline void print_by_ms(){
        for (uint32_t i = 0; i < t_comment.size(); i++){
          printf("[%s]%.2lf\t", t_comment[i].c_str(), get_cost_time_by_ms(t_list[i], t_list[i+1]));
        }
        printf("[total]%.2lf\n", get_cost_time_by_ms(t_list[0],t_list[t_comment.size()]));
      }
      inline void print_by_s(){
        for (uint32_t i = 0; i < t_comment.size(); i++){
          printf("[%s]%.2lf\t", t_comment[i].c_str(), get_cost_time_by_s(t_list[i], t_list[i+1]));
        }
        printf("[total]%.2lf\n", get_cost_time_by_s(t_list[0],t_list[t_comment.size()]));
      }
  };
};
#endif
