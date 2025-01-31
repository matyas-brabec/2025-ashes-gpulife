#include "GOL.hpp"

#include <cuda_runtime.h>
#include <iostream>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

#define MIN_NOF_NEIGH_FROM_ALIVE_TO_ALIVE 2
#define MAX_NOF_NEIGH_FROM_ALIVE_TO_ALIVE 3
#define MIN_NOF_NEIGH_FROM_DEAD_TO_ALIVE 3
#define MAX_NOF_NEIGH_FROM_DEAD_TO_ALIVE 3

#define ALIVE  1
#define DEAD   0


namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

template <typename policy, typename CELL_TYPE>
__device__ void setSubCellD (CELL_TYPE *cell, char pos, unsigned char subcell)
{
    constexpr std::size_t ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;
	
    CELL_TYPE mask = 0xFF;
	CELL_TYPE maskNewCell = subcell;
	
	// Erase pos content in cell:
	mask = mask << (ELEMENTS_PER_CELL - 1 - pos)*8;
	mask = ~mask;
	*cell = *cell & mask;
	
	// Add subcell content to cell in pos:
	*cell = *cell | (maskNewCell << (ELEMENTS_PER_CELL - 1 - pos)*8);
}

template <typename policy, typename CELL_TYPE>
__device__ unsigned char getSubCellD (CELL_TYPE cell, char pos)
{
    constexpr std::size_t ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;

	return (cell >> (ELEMENTS_PER_CELL - 1 - pos)*8);
}

template <typename policy, typename CELL_TYPE>
__global__ void GOL_packed (int GRID_SIZE, CELL_TYPE *grid, CELL_TYPE *newGrid, int *GPU_lookup_table)
{
    constexpr int ELEMENTS_PER_CELL = policy::ELEMENTS_PER_CELL;
    constexpr int CELL_NEIGHBOURS = policy::CELL_NEIGHBOURS;
    const int ROW_SIZE = GRID_SIZE / ELEMENTS_PER_CELL;

    // We want id ∈ [1,SIZE]
    const int iy = blockDim.y * blockIdx.y + threadIdx.y + 1;
    const int ix = blockDim.x * blockIdx.x + threadIdx.x + 1;
    const int id = iy * (ROW_SIZE+2) + ix;
    CELL_TYPE cell, new_cell=0; 
    CELL_TYPE up_cell, down_cell, right_cell, left_cell;                // Up,down,right,left cells
    CELL_TYPE upleft_cell, downleft_cell, upright_cell, downright_cell; // Diagonal cells
    unsigned char subcell;

    int numNeighbors;
    const int (*const lookup_table)[CELL_NEIGHBOURS+1] = (const int (*)[CELL_NEIGHBOURS+1]) GPU_lookup_table;

    if (iy>0 && iy <= GRID_SIZE && ix> 0 && ix <= ROW_SIZE) {
         cell = grid[id];

        // First (0) subcell:
        up_cell = grid[id-(ROW_SIZE+2)];
        down_cell = grid[id+(ROW_SIZE+2)];
        left_cell = grid[id+1];
        upleft_cell = grid[id-(ROW_SIZE+1)];
        downleft_cell = grid[id+(ROW_SIZE+3)];

        numNeighbors = getSubCellD<policy> (up_cell, 0) + getSubCellD<policy> (down_cell, 0); // upper lower
        numNeighbors += getSubCellD<policy> (left_cell, ELEMENTS_PER_CELL-1) + getSubCellD<policy> (cell, 1); // left right
        numNeighbors += getSubCellD<policy> (upleft_cell, ELEMENTS_PER_CELL-1) + getSubCellD<policy> (downleft_cell, ELEMENTS_PER_CELL-1); // diagonals left
        numNeighbors += getSubCellD<policy> (up_cell, 1) + getSubCellD<policy> (down_cell, 1); // diagonals right
        subcell = getSubCellD<policy> (cell, 0);
        setSubCellD<policy> (&new_cell, 0, lookup_table[subcell][numNeighbors]);

        // Middle subcells:
        for (int k=1; k<ELEMENTS_PER_CELL-1; k++) {
            numNeighbors = getSubCellD<policy> (up_cell, k) + getSubCellD<policy> (down_cell, k); // upper lower
            numNeighbors += getSubCellD<policy> (cell, k-1) + getSubCellD<policy> (cell, k+1); // left right
            numNeighbors += getSubCellD<policy> (up_cell, k-1) + getSubCellD<policy> (down_cell, k-1); // diagonals left
            numNeighbors += getSubCellD<policy> (up_cell, k+1) + getSubCellD<policy> (down_cell, k+1); // diagonals right
            subcell = getSubCellD<policy> (cell, k);
            setSubCellD<policy> (&new_cell, k, lookup_table[subcell][numNeighbors]);
        }

        // Last (ELEMENTS_PER_CELL-1) subcell:
        right_cell = grid[id-1];
        upright_cell = grid[id-(ROW_SIZE+3)];
        downright_cell = grid[id+(ROW_SIZE+1)];

        numNeighbors = getSubCellD<policy> (up_cell, ELEMENTS_PER_CELL-1) + getSubCellD<policy> (down_cell, ELEMENTS_PER_CELL-1); // upper lower
        numNeighbors += getSubCellD<policy> (cell, ELEMENTS_PER_CELL-2) + getSubCellD<policy> (right_cell, 0); // left right
        numNeighbors += getSubCellD<policy> (up_cell, ELEMENTS_PER_CELL-2) + getSubCellD<policy> (down_cell, ELEMENTS_PER_CELL-2); // diagonals left
        numNeighbors += getSubCellD<policy> (upright_cell, 0) + getSubCellD<policy> (downright_cell, 0); // diagonals right
        subcell = getSubCellD<policy> (cell, ELEMENTS_PER_CELL-1);
        setSubCellD<policy> (&new_cell, ELEMENTS_PER_CELL-1, lookup_table[subcell][numNeighbors]);


        // Copy new_cell to newGrid:
        newGrid[id] = new_cell;

/* 
        // Get the number of neighbors for a given grid point
        numNeighbors = grid[id+(SIZE+2)] + grid[id-(SIZE+2)] //upper lower
                     + grid[id+1] + grid[id-1]             //right left
                     + grid[id+(SIZE+3)] + grid[id-(SIZE+3)] //diagonals
                     + grid[id-(SIZE+1)] + grid[id+(SIZE+1)];
 
        CELL_TYPE cell = grid[id];
        newGrid[id] = lookup_table[cell][numNeighbors];
*/

    }
}

__global__ void kernel_init_lookup_table (int *GPU_lookup_table) {
    constexpr std::size_t CELL_NEIGHBOURS = 8;

    int (*lookup_table)[CELL_NEIGHBOURS+1] = (int (*)[CELL_NEIGHBOURS+1]) GPU_lookup_table;

    if ( threadIdx.y < 2 && threadIdx.x < (CELL_NEIGHBOURS+1) ) {
        // Init lookup_table for GOL
        // Classic B3S23 GOL:
	    //lookup_table[cases] = {
		//    d,d,d,a, d,d,d,d,d // DEAD is current state
		//    d,d,a,a, d,d,d,d,d // ALIVE is current state
		    if (threadIdx.y==0) 
                if (threadIdx.x >= MIN_NOF_NEIGH_FROM_DEAD_TO_ALIVE && threadIdx.x <= MAX_NOF_NEIGH_FROM_DEAD_TO_ALIVE)
			        lookup_table[threadIdx.y][threadIdx.x] = ALIVE;
		        else
			        lookup_table[threadIdx.y][threadIdx.x] = DEAD; 
 
		    if (threadIdx.y==1) 
                if (threadIdx.x >= MIN_NOF_NEIGH_FROM_ALIVE_TO_ALIVE && threadIdx.x <= MAX_NOF_NEIGH_FROM_ALIVE_TO_ALIVE)
			        lookup_table[threadIdx.y][threadIdx.x] =  ALIVE;
		        else
			        lookup_table[threadIdx.y][threadIdx.x] = DEAD;  
    }
}

} // namespace

template <typename grid_cell_t, typename policy>
void GOL_Packed_sota<grid_cell_t, policy>::run_kernel(size_type iterations) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGridx = (int)ceil(ROW_SIZE/(float)BLOCK_SIZE);
    int linGridy = (int)ceil(GRID_SIZE/(float)BLOCK_SIZE);
    dim3 gridSize(linGridx,linGridy,1);
 
    infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
    _performed_iterations = this->params.iterations;

    for (std::size_t i = 0; i < iterations; ++i) {
        if (stop_watch.time_is_up()) {
            _performed_iterations = i;
            break;
        }
        
        if (i != 0) {
            std::swap(grid, new_grid);
        }

        GOL_packed<policy><<<gridSize, blockSize>>>(GRID_SIZE, grid, new_grid, GPU_lookup_table);
        CUCH(cudaPeekAtLastError());
    }
}

template <typename grid_cell_t, typename policy>
void GOL_Packed_sota<grid_cell_t, policy>::init_lookup_table() {
    dim3 blockSize(32, 32, 1);

    kernel_init_lookup_table<<<1,blockSize>>>(GPU_lookup_table);
}


} // namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_sota<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_64_bit_policy>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_sota<common::INT, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_64_bit_policy>;

template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_sota<common::CHAR, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy>;
template class algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::GOL_Packed_sota<common::INT, algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU::_32_bit_policy>;
