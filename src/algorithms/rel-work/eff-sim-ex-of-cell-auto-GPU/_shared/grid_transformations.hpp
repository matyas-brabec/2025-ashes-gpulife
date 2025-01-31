#ifndef rel_work_GRID_TRANSFORMATIONS_HPP
#define rel_work_GRID_TRANSFORMATIONS_HPP

#include "../../../../infrastructure/grid.hpp"

namespace algorithms::rel_work::eff_sim_ex_of_cell_auto_GPU {

class GridTransformer {

  template <typename cell_t>
  using DataGrid = infrastructure::Grid<2, cell_t>;
  
  public:

    template <typename cell_t_to, typename cell_t_from, std::size_t halo_size = 1>
    static DataGrid<cell_t_to> transform_grid_with_halo(const DataGrid<cell_t_from>& data) {
        DataGrid<cell_t_to> result(
            data.template size_in<0>() + 2 * halo_size,
            data.template size_in<1>() + 2
        );

        auto size = result.size();
        auto raw_data = result.data();

        for (size_t i = 0; i < size; ++i) {
            raw_data[i] = 0;
        }

        for (size_t y = 0; y < data.template size_in<1>(); ++y) {
            for (size_t x = 0; x < data.template size_in<0>(); ++x) {
                result[x + halo_size][y + 1] = static_cast<cell_t_to>(data[x][y]);
            }
        }

        return result;
    }

    template <typename cell_t_to, typename cell_t_from, std::size_t halo_size = 1>
    static DataGrid<cell_t_to> transform_grid_remove_halo(const DataGrid<cell_t_from>& data) {
        DataGrid<cell_t_to> result(
            data.template size_in<0>() - 2 * halo_size,
            data.template size_in<1>() - 2
        );

        for (size_t y = 0; y < result.template size_in<1>(); ++y) {
            for (size_t x = 0; x < result.template size_in<0>(); ++x) {
                result[x][y] = static_cast<cell_t_to>(data[x + halo_size][y + 1]);
            }
        }

        return result;
    }
};

}

#endif // rel_work_GRID_TRANSFORMATIONS_HPP