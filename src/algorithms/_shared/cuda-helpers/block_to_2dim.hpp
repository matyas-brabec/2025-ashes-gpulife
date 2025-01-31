#ifndef ALGORITHMS_SHARED_CUDA_HELPERS_BLOCK_TO_2DIM_HPP
#define ALGORITHMS_SHARED_CUDA_HELPERS_BLOCK_TO_2DIM_HPP

#include <cstddef>
#include <cuda_runtime.h>
#include <stdexcept>

namespace algorithms {

template <typename num_t>
num_t log_2(num_t x) {
    num_t result = 0;
    while (x >>= 1) {
        result++;
    }
    return result;
}

template <typename num_t>
dim3 get_2d_block(num_t block_size) {
    if (block_size == 0) {
        throw std::invalid_argument("Block size must be greater than 0");
    }

    auto l = log_2(block_size);

    std::size_t one = 1;

    std::size_t x, y;

    if (l % 2 == 0) {
        x = one << (l / 2);
    } 
    else {
        x = one << (l / 2 + 1);
    }
    
    y = one << (l / 2);
    
    return {static_cast<unsigned int>(x), static_cast<unsigned int>(y)};
}

}

#endif // ALGORITHMS_SHARED_CUDA_HELPERS_BLOCK_TO_2DIM_HPP