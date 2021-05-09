#ifndef ZQ_CPP_LIB_MAX_H
#define ZQ_CPP_LIB_MAX_H

#include <stdint.h>
#include <algorithm>

#include <stdio.h>

// namespace zq_cpp_lib{
//     namespace high_perf_algo{
//         template <typename T>
//         inline T max(T x, T y){
//             return std::max(x,y);
//         }
//         template<>
//         inline int32_t max<int32_t>(int32_t x, int32_t y){
//             printf("x=%d, y=%d\n", x, y);
//             return y&((x-y)>>31)|x&~((x-y)>>31);
//         } 
//         template<>
//         inline int64_t max<int64_t>(int64_t x, int64_t y){
//             return y&((x-y)>>63)|x&~((x-y)>>63);
//         } 
//         // template<>
//         // inline float max<float>(float x, float y){
//         //     (reinterpret_cast<int32_t*>(&x))[0]&=(1UL<<31)-1;
//         //     return x;
//         // } 
//         // template<>
//         // inline double max<double>(double x, double y){
//         //     (reinterpret_cast<int64_t*>(&x))[0]&=(1ULL<<63)-1;
//         //     return x;
//         // }
        
//     }
// }


#endif
