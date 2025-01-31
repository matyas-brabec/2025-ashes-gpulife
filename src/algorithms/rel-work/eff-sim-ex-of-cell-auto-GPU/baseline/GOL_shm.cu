#include "GOL.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

template <int BLOCK_x, int BLOCK_y>
__global__ void GOL_shm(int dim, int *grid, int *newGrid)
{
        int iy = (blockDim.y -2) * blockIdx.y + threadIdx.y;
        int ix = (blockDim.x -2) * blockIdx.x + threadIdx.x;
        int id = iy * (dim+2) + ix;
 
        int i = threadIdx.y;
        int j = threadIdx.x;
        int numNeighbors;
 
        // Declare the shared memory on a per block level
        __shared__ int s_grid[BLOCK_y][BLOCK_x];
 
       // Copy cells into shared memory
       if (ix <= dim+1 && iy <= dim+1)
           s_grid[i][j] = grid[id];
 
       //Sync all threads in block
        __syncthreads();
 
       if (iy <= dim && ix <= dim) {
           if(i != 0 && i !=blockDim.y-1 && j != 0 && j !=blockDim.x-1) {
 
               // Get the number of neighbors for a given grid point
               numNeighbors = s_grid[i+1][j] + s_grid[i-1][j] //upper lower
                            + s_grid[i][j+1] + s_grid[i][j-1] //right left
                            + s_grid[i+1][j+1] + s_grid[i-1][j-1] //diagonals
                            + s_grid[i-1][j+1] + s_grid[i+1][j-1];
 
                int cell = s_grid[i][j];
 
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
}

} // namespace

template <typename grid_cell_t, BaselineVariant variant>
template <int BLOCK_SIZE_x, int BLOCK_SIZE_y>
void GOL_Baseline<grid_cell_t, variant>::run_kernel_shm_specialized(size_type iterations) {
    dim3 blockSize(BLOCK_SIZE_x, BLOCK_SIZE_y,1);
    int linGrid_x = (int)ceil(dim/(float)(BLOCK_SIZE_x-2));
    int linGrid_y = (int)ceil(dim/(float)(BLOCK_SIZE_y-2));
    dim3 gridSize(linGrid_x,linGrid_y,1);

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

        GOL_shm<BLOCK_SIZE_x, BLOCK_SIZE_y><<<gridSize, blockSize>>>(dim, grid, new_grid);
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
