#ifndef GOL_CPU_BITWISE_COLS_HPP
#define GOL_CPU_BITWISE_COLS_HPP

#include "../../infrastructure/algorithm.hpp"
#include "../_shared/bitwise/bit_word_types.hpp"
#include "../_shared/bitwise/general_bit_grid.hpp"
#include "../_shared/bitwise/bitwise-ops/templated-cols.hpp"
#include <cstddef>
#include <memory>

namespace algorithms {

template <typename grid_cell_t, std::size_t Bits, template <typename word_type> class BitOps>
class GoLCpuBitwise : public infrastructure::Algorithm<2, grid_cell_t> {
  public:
    using word_type = typename BitsConst<Bits>::word_type;
    using BitGrid = algorithms::GeneralBitGrid<word_type, typename BitOps<word_type>::bit_grid_mode>;
    using DataGrid = infrastructure::Grid<2, grid_cell_t>;
    using size_type = BitGrid::size_type;

    void set_and_format_input_data(const DataGrid& data) override {
        bit_grid = std::make_unique<BitGrid>(data);

        original_x_size = data.template size_in<0>();
        original_y_size = data.template size_in<1>();
    }

    void initialize_data_structures() override {
        intermediate_bit_grid = std::make_unique<BitGrid>(original_x_size, original_y_size);
    }

    void run(size_type iterations) override {
        auto x_size = bit_grid->x_size();
        auto y_size = bit_grid->y_size();

        auto source = bit_grid.get();
        auto target = intermediate_bit_grid.get();

        infrastructure::StopWatch stop_watch(this->params.max_runtime_seconds);
        _performed_iterations = this->params.iterations;

        for (size_type i = 0; i < iterations; ++i) {
            if (stop_watch.time_is_up()) {
                _performed_iterations = i;
                break;
            }

            for (size_type y = 0; y < y_size; ++y) {
                for (size_type x = 0; x < x_size; ++x) {
                    // clang-format off

                    word_type lt, ct, rt;
                    word_type lc, cc, rc;
                    word_type lb, cb, rb;

                    load(source, x, y,
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb);

                    word_type new_center = BitOps<word_type>::compute_center_word(
                        lt, ct, rt,
                        lc, cc, rc,
                        lb, cb, rb
                    );

                    target->set_word(x, y, new_center);

                    // clang-format on
                }
            }

            std::swap(source, target);
        }
        final_bit_grid = source;
    }

    void finalize_data_structures() override {
    }

    DataGrid fetch_result() override {
        return final_bit_grid->template to_grid<grid_cell_t>();
    }

    std::size_t actually_performed_iterations() const override {
        return _performed_iterations;
    }

  private:
    // clang-format off
    void load(const BitGrid* grid, size_type x, size_type y,
        word_type& lt, word_type& ct, word_type& rt,
        word_type& lc, word_type& cc, word_type& rc,
        word_type& lb, word_type& cb, word_type& rb) {

        load_one(grid, lt, x - 1, y - 1);
        load_one(grid, ct, x,     y - 1);
        load_one(grid, rt, x + 1, y - 1);
        
        load_one(grid, lc, x - 1, y    );
        load_one(grid, cc, x,     y    );
        load_one(grid, rc, x + 1, y    );
        
        load_one(grid, lb, x - 1, y + 1);
        load_one(grid, cb, x,     y + 1);
        load_one(grid, rb, x + 1, y + 1);
    }
    // clang-format on

    void load_one(const BitGrid* grid, word_type& word, size_type x, size_type y) {
        auto x_size = grid->x_size();
        auto y_size = grid->y_size();

        if (x < 0 || x >= x_size || y < 0 || y >= y_size) {
            word = 0;
        }
        else {
            word = grid->get_word(x, y);
        }
    }

    size_type original_x_size;
    size_type original_y_size;

    DataGrid _result;

    std::unique_ptr<BitGrid> bit_grid;
    std::unique_ptr<BitGrid> intermediate_bit_grid;

    BitGrid* final_bit_grid;

    std::size_t _performed_iterations;
};

} // namespace algorithms

#endif // GOL_CPU_BITWISE_COLS_HPP