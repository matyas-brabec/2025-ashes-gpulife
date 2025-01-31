#ifndef TIMER_HPP
#define TIMER_HPP

#include "../algorithms/_shared/cuda-helpers/cuch.hpp"
#include <chrono>
#include <cuda_runtime.h>

namespace infrastructure {

class Timer {
  public:
    using clock_type = std::chrono::high_resolution_clock;

    template <typename Func>
    static double measure_ms(Func&& func) {

        CUCH(cudaDeviceSynchronize());
        auto start = clock_type::now();
        func();
        
        CUCH(cudaDeviceSynchronize());
        auto end = clock_type::now();

        double nano_sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return nano_sec / 1e6;
    }
};

class StopWatch {
  public:
    using clock_type = std::chrono::high_resolution_clock;
    static constexpr long MAX_TIME = std::numeric_limits<long>::max();

    StopWatch(std::size_t count_down_seconds) : count_down_seconds(count_down_seconds) {
        restart();
    }

    void restart() {
        start_time = clock_type::now();
    }

    bool time_is_up() const {
        const auto now = clock_type::now();
        const auto elapsed_secs = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        return elapsed_secs >= count_down_seconds;
    }

  private:
    long count_down_seconds;
    std::chrono::time_point<clock_type> start_time;
};


} // namespace infrastructure

#endif // TIMER_HPP