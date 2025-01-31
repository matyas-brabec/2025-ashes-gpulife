#ifndef CUDA_MEMCPY_BENCH_KERNEL_CU
#define CUDA_MEMCPY_BENCH_KERNEL_CU

#include "cuda_memcpy_bench.hpp"
#include "../cuda-naive/models.hpp"
#include <cuda_runtime.h>
#include "../../infrastructure/timer.hpp"
#include "../_shared/common_grid_types.hpp"

namespace algorithms {

using idx_t = std::int64_t;

namespace {

template <typename grid_cell_t>
__global__ void memcpy_kernel(NaiveGridOnCuda<grid_cell_t> data) {
    idx_t x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x >= data.x_size)
        return;

    idx_t idx = x;

    data.output[idx] = data.input[idx];
}

} // namespace

template <typename grid_cell_t>
void CudaMemcpy<grid_cell_t>::run_kernel(size_type iterations) {
    std::size_t block_size = this->params.thread_block_size;
    std::size_t grid_size = (cuda_data.x_size + block_size - 1) / block_size;

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }
        
        if (i != 0) {
            std::swap(cuda_data.input, cuda_data.output);
        }

        memcpy_kernel<<<grid_size, block_size>>>(cuda_data);
        CUCH(cudaPeekAtLastError());
    }
}

} // namespace algorithms

template class algorithms::CudaMemcpy<common::CHAR>;
template class algorithms::CudaMemcpy<common::INT>;

#endif // CUDA_NAIVE_KERNEL_CU