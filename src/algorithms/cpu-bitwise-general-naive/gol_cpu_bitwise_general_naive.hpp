#ifndef GOL_CPU_BITWISE_NAIVE_HPP
#define GOL_CPU_BITWISE_NAIVE_HPP

#include "../../debug_utils/pretty_print.hpp"
#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include <chrono>
#include <cstddef>
#include <iostream>
#include <thread>

namespace algorithms {

template <typename grid_cell_t, std::size_t Bits, typename bit_grid_mode>
class GoLCpuBitwiseNaive : public infrastructure::Algorithm<2, grid_cell_t> {
  public:
    using size_type = std::size_t;
    using word_type = typename BitsConst<Bits>::word_type;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using BitGrid = GeneralBitGrid<word_type, bit_grid_mode>;

    void set_and_format_input_data(const DataGrid& data) override {
        _initial_source_bit_grid = std::make_unique<BitGrid>(data);

        x_size = data.template size_in<0>();
        y_size = data.template size_in<1>();
    }

    void initialize_data_structures() override {
        _intermediate = std::make_unique<BitGrid>(x_size, y_size);
    }

    void run(size_type iterations) override {
        BitGrid* source = _initial_source_bit_grid.get();
        BitGrid* target = _intermediate.get();


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
                    auto cell_state = source->get_value_at(x, y);

                    if (cell_state == BitGrid::ALIVE) {
                        if (alive_neighbours < 2 || alive_neighbours > 3) {
                            target->set_value_at(x, y, BitGrid::DEAD);
                        }
                        else {
                            target->set_value_at(x, y, BitGrid::ALIVE);
                        }
                    }
                    else {
                        if (alive_neighbours == 3) {
                            target->set_value_at(x, y, BitGrid::ALIVE);
                        }
                        else {
                            target->set_value_at(x, y, BitGrid::DEAD);
                        }
                    }
                }
            }

            std::swap(target, source);
        }

        ptr_to_result_bit_grid = source;
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return ptr_to_result_bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    size_type count_alive_neighbours(const BitGrid& grid, size_type x, size_type y) {
        size_type alive_neighbours = 0;

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

                auto state = grid.get_value_at(x_neighbour, y_neighbour);

                if (state == BitGrid::ALIVE) {
                    ++alive_neighbours;
                }
            }
        }

        return alive_neighbours;
    }

    std::unique_ptr<BitGrid> _initial_source_bit_grid;
    std::unique_ptr<BitGrid> _intermediate;

    BitGrid* ptr_to_result_bit_grid = nullptr;

    std::size_t x_size;
    std::size_t y_size;

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CPU_NAIVE_HPP