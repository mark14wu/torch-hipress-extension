#ifndef ZQ_CPP_LIB_ABS_H
#define ZQ_CPP_LIB_ABS_H

#include <stdint.h>
#include <cstdlib>

namespace zq_cpp_lib{
    namespace high_perf_algo{
        template <typename T>
        inline T abs(T x){
            return std::abs(x);
        }
        template<>
        inline int32_t abs<int32_t>(int32_t x){
            int32_t y = x>>31;
            return (x+y)^y;
        } 
        template<>
        inline int64_t abs<int64_t>(int64_t x){
            int64_t y = x>>63;
            return (x+y)^y;
        } 
        template<>
        inline float abs<float>(float x){
            (reinterpret_cast<int32_t*>(&x))[0]&=(1UL<<31)-1;
            return x;
        } 
        template<>
        inline double abs<double>(double x){
            (reinterpret_cast<int64_t*>(&x))[0]&=(1ULL<<63)-1;
            return x;
        }
        
    }
}


#endif
