#include "../time_cost.hpp"
#include <unistd.h>

int main(){
    zq_cpp_lib::time_cost t;
    t.start();
    usleep(10);
    t.record("usleep(10)");
    usleep(100);
    t.record("usleep(100)");
    usleep(1000);
    t.record("usleep(1000)");
    usleep(10000);
    t.record("usleep(10000)");
    usleep(100000);
    t.record("usleep(100000)");
    t.print_by_us();
    return 0;
}