#ifndef ZQ_CPP_LIB_NAIVE_RANDOM_HPP
#define ZQ_CPP_LIB_NAIVE_RANDOM_HPP

#include <stdint.h>

#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#else
#define CUDA_HOSTDEV
#endif

namespace zq_cpp_lib{

typedef uint32_t RANDOM_INT;
class naive_random{
private:
    RANDOM_INT _a=1103515245;
    RANDOM_INT _c=12345;
    RANDOM_INT _x;
protected:
    RANDOM_INT _max = ~0;
    double _max_double = static_cast<double>(_max);
public:
    CUDA_HOSTDEV
    void inline srand(const RANDOM_INT seed){
        _x = seed;
        rand();
    }
    CUDA_HOSTDEV
    RANDOM_INT rand(){
        _x = _a*_x + _c;
        return _x;
    }
};

template <typename T>
class naive_int_random: public naive_random{
private:
    T _lowerbound, _upperbound, _boundrange;
public:
    CUDA_HOSTDEV
    naive_int_random(T lowerbound_=0, T upperbound_=128){
        _lowerbound = lowerbound_;
        _upperbound = upperbound_;
        _boundrange = _upperbound - _lowerbound + 1;
    }
    CUDA_HOSTDEV
    T operator()(){
        return (static_cast<T>(rand())%_boundrange)+_lowerbound;
    }
};

template <typename T>
class naive_real_random: public naive_random{
private:
    T _lowerbound, _upperbound, _boundrange;
public:
    CUDA_HOSTDEV
    naive_real_random(T lowerbound_=0.0, T upperbound_=1.0){
        _lowerbound = lowerbound_;
        _upperbound = upperbound_;
        _boundrange = _upperbound - _lowerbound;
        _max_double = _max_double/_boundrange;
    }
    CUDA_HOSTDEV
    T operator()(){
        return static_cast<T>(static_cast<double>(rand())/_max_double);
    }
};

}

#endif