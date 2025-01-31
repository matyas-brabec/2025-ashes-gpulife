#ifndef CUDA_NAIVE_MODELS_HPP
#define CUDA_NAIVE_MODELS_HPP

#include <cstddef>
namespace algorithms {

template <typename grid_cell_t>
struct NaiveGridOnCuda {
    grid_cell_t* input;
    grid_cell_t* output;
    std::size_t x_size;
    std::size_t y_size;
};

} // namespace algorithms

#endif // CUDA_NAIVE_MODELS_HPP