#ifndef GOL_CPU_NAIVE_HPP
#define GOL_CPU_NAIVE_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include <chrono>
#include <iostream>
#include <thread>

namespace algorithms {

template <typename grid_cell_t>
class GoLCpuNaive : public infrastructure::Algorithm<2, grid_cell_t> {
  public:
    using size_type = std::size_t;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;

    void set_and_format_input_data(const DataGrid& data) override {
        _result = data;
    }

    void initialize_data_structures() override {
        _intermediate = _result;
    }

    void run(size_type iterations) override {
        DataGrid* source = &_result;
        DataGrid* target = &_intermediate;

        auto x_size = _result.template size_in<0>();
        auto y_size = _result.template size_in<1>();

        if (this->params.animate_output) {
            print(*source, 0);
        }

        infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
        _performed_iterations = this->params.iterations;

        for (size_type i = 0; i < iterations; ++i) {
            if (stop_watch.time_is_up()) {
                _performed_iterations = i;
                break;
            }

            for (size_type y = 0; y < y_size; ++y) {
                for (size_type x = 0; x < x_size; ++x) {

                    auto alive_neighbours = count_alive_neighbours(*source, x, y);

                    if ((*source)[x][y] == 1) {
                        if (alive_neighbours < 2 || alive_neighbours > 3) {
                            (*target)[x][y] = 0;
                        }
                        else {
                            (*target)[x][y] = 1;
                        }
                    }
                    else {
                        if (alive_neighbours == 3) {
                            (*target)[x][y] = 1;
                        }
                        else {
                            (*target)[x][y] = 0;
                        }
                    }
                }
            }

            if (this->params.animate_output) {
                print(*target, i + 1);
            }

            std::swap(target, source);
        }

        _result = *source;
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return std::move(_result);
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    size_type count_alive_neighbours(const DataGrid& grid, size_type x, size_type y) {
        size_type alive_neighbours = 0;

        size_type x_size = grid.template size_in<0>();
        size_type y_size = grid.template size_in<1>();

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                // Skip the cell itself
                if (i == 0 && j == 0)
                    continue;

                auto x_neighbour = x + i;
                auto y_neighbour = y + j;

                constexpr std::size_t zero = 0;

                if (x_neighbour < zero || x_neighbour >= x_size || y_neighbour < zero || y_neighbour >= y_size)
                    continue;

                alive_neighbours += grid[x_neighbour][y_neighbour] > 0 ? 1 : 0;
            }
        }

        return alive_neighbours;
    }

    void move_cursor_up_left(const DataGrid& grid) {
        std::cout << "\033[" << grid.template size_in<0>() + 2 << "A";
        std::cout << "\033[" << grid.template size_in<1>() << "D";
    }

    void print(const DataGrid& grid, size_type iter) const {

        std::cout << "Iteration: " << iter << std::endl;

        std::cout << debug_utils::pretty_g<2, grid_cell_t>(grid) << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    DataGrid _result;
    DataGrid _intermediate;

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP