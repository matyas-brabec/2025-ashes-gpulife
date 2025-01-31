#include "GOL.hpp"

#include <cuda_runtime.h>

#include "../../../../infrastructure/timer.hpp"
#include "../../../_shared/common_grid_types.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

namespace {

template <typename grid_cell_t>
__global__ void GOL(int dim, cudaTextureObject_t gridTex, int *newGrid)
{
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int id = iy * dim + ix;
 
    int numNeighbors;
 
    float iyTex = (iy + 0.5f)/dim;
    float ixTex = (ix + 0.5f)/dim;
    float oneTex = 1.0f/dim;
 
    if(iy < dim && ix < dim)
    {
        //Get the number of neighbors for a given grid point
        numNeighbors = tex2D<grid_cell_t>(gridTex, iyTex+oneTex, ixTex) //upper/lower
                    + tex2D<grid_cell_t>(gridTex, iyTex-oneTex, ixTex)
                    + tex2D<grid_cell_t>(gridTex, iyTex, ixTex+oneTex) //right/left
                    + tex2D<grid_cell_t>(gridTex, iyTex, ixTex-oneTex)
                    + tex2D<grid_cell_t>(gridTex, iyTex-oneTex, ixTex-oneTex) //diagonals
                    + tex2D<grid_cell_t>(gridTex, iyTex-oneTex, ixTex+oneTex)
                    + tex2D<grid_cell_t>(gridTex, iyTex+oneTex, ixTex-oneTex) 
                    + tex2D<grid_cell_t>(gridTex, iyTex+oneTex, ixTex+oneTex);
    
        int cell = tex2D<grid_cell_t>(gridTex, iyTex, ixTex);
    
        //Here we have explicitly all of the game rules
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
void GOL_Baseline<grid_cell_t, variant>::run_kernel_texture(size_type iterations) {
    auto BLOCK_SIZE = get_2d_block(this->params.thread_block_size).x;

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE,1);
    int linGrid = (int)ceil(dim/(float)BLOCK_SIZE);
    dim3 gridSize(linGrid,linGrid,1);

    int width = dim;
    int height = dim;

    cudaChannelFormatKind kind = cudaChannelFormatKindNone;

    if constexpr (std::is_same_v<grid_cell_t, common::INT>) {
        kind = cudaChannelFormatKindSigned;
    } else {
        throw std::runtime_error("Unsupported grid cell type");
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, kind);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    cudaMemcpy2DToArray(cuArray, 0, 0, grid, width * sizeof(grid_cell_t), width * sizeof(grid_cell_t), height, cudaMemcpyDefault);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t gridTex = 0;
    cudaCreateTextureObject(&gridTex, &resDesc, &texDesc, NULL);

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

        GOL<grid_cell_t><<<gridSize, blockSize>>>(dim, gridTex, new_grid);
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

