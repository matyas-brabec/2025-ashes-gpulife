#include "GOL.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

__global__ void GOL_basic(int dim, int *grid, int *newGrid)
{
    // We want id ∈ [1,dim]
    int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int id = iy * (dim+2) + ix;
 
    int numNeighbors;
 
    if (iy <= dim && ix <= dim) {
 
        // Get the number of neighbors for a given grid point
        numNeighbors = grid[id+(dim+2)] + grid[id-(dim+2)] //upper lower
                     + grid[id+1] + grid[id-1]             //right left
                     + grid[id+(dim+3)] + grid[id-(dim+3)] //diagonals
                     + grid[id-(dim+1)] + grid[id+(dim+1)];
 
        int cell = grid[id];
        // Here we have explicitly all of the game rules
        if (cell == 1 && numNeighbors < 2)
            newGrid[id] = 0;
        else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
            newGrid[id] = 1;
        else if (cell == 1 && numNeighbors > 3)
            newGrid[id] = 0;
        else if (cell == 0 && numNeighbors == 3)
            newGrid[id] = 1;
        else
            newGrid[id] = cell;
    }
}

} // namespace

template <typename grid_cell_t, BaselineVariant variant>
void GOL_Baseline<grid_cell_t, variant>::run_kernel_basic(size_type iterations) {
    auto BLOCK_SIZE = get_2d_block(this->params.thread_block_size).x;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
    dim3 gridSize(linGrid,linGrid,1);

    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (size_type i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }

        if (i != 0) {
            std::swap(grid, new_grid);
        }

        GOL_basic<<<gridSize, blockSize>>>(dim, grid, new_grid);
        CUCH(cudaPeekAtLastError());
    }
}

}

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::Basic>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::SharedMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::TextureMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::Basic>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::SharedMemory>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Baseline<common::INT,  algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::BaselineVariant::TextureMemory>;
