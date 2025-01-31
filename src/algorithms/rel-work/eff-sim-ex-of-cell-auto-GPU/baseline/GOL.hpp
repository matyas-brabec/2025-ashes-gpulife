#ifndef rel_work_GOL_HPP
#define rel_work_GOL_HPP

#include "../../../../infrastructure/algorithm.hpp"
#include "../_shared/grid_transformations.hpp"
#include <cstddef>
#include "../../../_shared/cuda-helpers/block_to_2dim.hpp"
#include <iostream>

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

enum class BaselineVariant {
    Basic = 0,
    SharedMemory = 1,
    TextureMemory = 2,
};

template <typename grid_cell_t, BaselineVariant variant>
class GOL_Baseline : public infrastructure::Algorithm<2, grid_cell_t> {
  public:

    using size_type = std::size_t;

    template <typename cell_t>
    using DataGrid = infrastructure::Grid<2, cell_t>;

    void set_and_format_input_data(const DataGrid<grid_cell_t>& data) override {
        input_output_data_grid = GridTransformer::transform_grid_with_halo<int>(data);
    }

    void initialize_data_structures() override {
        auto bytes = sizeof(int)*(dim+2)*(dim+2);

        CUCH(cudaMalloc(&grid, bytes));
        CUCH(cudaMalloc(&new_grid, bytes));

        CUCH(cudaMemcpy(grid, input_output_data_grid.data(), bytes, cudaMemcpyHostToDevice));
    }

    void run(size_type iterations) override {
        if constexpr (variant == BaselineVariant::Basic) {
            run_kernel_basic(iterations);
        } 
        else if constexpr (variant == BaselineVariant::SharedMemory) {
            run_kernel_shm(iterations);
        } 
        else if constexpr (variant == BaselineVariant::TextureMemory) {
            run_kernel_texture(iterations);
        }
        else {
            throw std::runtime_error("Unknown variant");
        }
    }

    void finalize_data_structures() override {
        auto bytes = sizeof(int)*(dim+2)*(dim+2);

        CUCH(cudaMemcpy(input_output_data_grid.data(), new_grid, bytes, cudaMemcpyDeviceToHost));

        CUCH(cudaFree(grid));
        CUCH(cudaFree(new_grid));
    }

    DataGrid<grid_cell_t> fetch_result() override {
        return GridTransformer::transform_grid_remove_halo<grid_cell_t>(input_output_data_grid);
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

    void set_params(const infrastructure::ExperimentParams& params) override {
        this->params = params;
        dim = params.grid_dimensions[0];

        if (params.grid_dimensions[0] != params.grid_dimensions[1]) {
            throw std::runtime_error("Only square grids are supported");
        }

        if (variant == BaselineVariant::SharedMemory) {
            auto block = get_2d_block(this->params.thread_block_size);

            if (block.x != block.y) {
                throw std::runtime_error("Only square thread blocks are supported");
            }
        }

    }

  private:
    int* grid;
    int* new_grid;
    int dim;

    size_type _performed_iterations;

    DataGrid<int> input_output_data_grid;

    template <int BLOCK_SIZE_x, int BLOCK_SIZE_y>
    void run_kernel_shm_specialized(size_type iterations);
    
    void run_kernel_basic(size_type iterations);
    
    void run_kernel_texture(size_type iterations);


    void run_kernel_shm(size_type iterations) {
        auto block = get_2d_block(this->params.thread_block_size);

        if (block.x == 8) {
            if (block.y == 8) {
                run_kernel_shm_specialized<8, 8>(iterations);
            }
            else if (block.y == 16) {
                run_kernel_shm_specialized<8, 16>(iterations);
            }
            else if (block.y == 32) {
                run_kernel_shm_specialized<8, 32>(iterations);
            }
            else {
                throw std::runtime_error("Unsupported block size");
            }
        }
        else if (block.x == 16) {
            if (block.y == 8) {
                run_kernel_shm_specialized<16, 8>(iterations);
            }
            else if (block.y == 16) {
                run_kernel_shm_specialized<16, 16>(iterations);
            }
            else if (block.y == 32) {
                run_kernel_shm_specialized<16, 32>(iterations);
            }
            else {
                throw std::runtime_error("Unsupported block size");
            }
        }
        else if (block.x == 32) {
            if (block.y == 8) {
                run_kernel_shm_specialized<32, 8>(iterations);
            }
            else if (block.y == 16) {
                run_kernel_shm_specialized<32, 16>(iterations);
            }
            else if (block.y == 32) {
                run_kernel_shm_specialized<32, 32>(iterations);
            }
            else {
                throw std::runtime_error("Unsupported block size");
            }
        }
        else {
            throw std::runtime_error("Unsupported block size");
        }
    }

};

}

#endif // rel_work_GOL_HPP