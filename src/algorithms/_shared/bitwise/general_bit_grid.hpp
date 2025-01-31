#ifndef GOL_BIT_COL_GRID_HPP
#define GOL_BIT_COL_GRID_HPP

#include <bitset>
#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../../debug_utils/pretty_print.hpp"
#include "../../../infrastructure/grid.hpp"
#include "./bit_modes.hpp"

using namespace debug_utils;

namespace algorithms {


template <typename word_type, typename bit_grid_mode>
class GeneralBitGrid {
  public:
    using size_type = std::size_t;

    using policy = typename bit_grid_mode::template policy<word_type>;

    template <typename grid_cell_t>
    using Grid = infrastructure::Grid<2, grid_cell_t>;

    using ONE_CELL_STATE = bool;
    constexpr static ONE_CELL_STATE DEAD = 0;
    constexpr static ONE_CELL_STATE ALIVE = 1;

    GeneralBitGrid(size_type original_x_size, size_t original_y_size)
        : _x_size(original_x_size), _y_size(original_y_size) {
        _x_size = original_x_size / policy::X_BITS;
        _y_size = original_y_size / policy::Y_BITS;

        words_grid.resize(x_size() * y_size(), 0);
    }

    template <typename grid_cell_t>
    GeneralBitGrid(const Grid<grid_cell_t>& grid) {
        assert_dim_has_correct_size(grid);

        _x_size = grid.template size_in<0>() / policy::X_BITS;
        _y_size = grid.template size_in<1>() / policy::Y_BITS;

        words_grid.resize(x_size() * y_size(), 0);
        fill_grid(grid);
    }

  public:
    ONE_CELL_STATE get_value_at(std::size_t x, std::size_t y) const {
        word_type word = get_word_from_original_coords(x, y);
        auto bit_mask = policy::get_bit_mask_for(x, y);

        return (word & bit_mask) ? ALIVE : DEAD;
    }

    void set_value_at(std::size_t x, std::size_t y, ONE_CELL_STATE state) {
        auto word = get_word_from_original_coords(x, y);
        
        auto bit_mask = policy::get_bit_mask_for(x, y);

        if (state == ALIVE) {
            word |= bit_mask;
        }
        else {
            word &= ~bit_mask;
        }
        
        set_word_from_original_coords(x, y, word);
    }

    word_type get_word_from_original_coords(std::size_t x, std::size_t y) const {
        return get_word(x / policy::X_BITS, y / policy::Y_BITS);
    }

    word_type get_word(std::size_t x, std::size_t y) const {
        return words_grid[idx(x, y)];
    }

    void set_word_from_original_coords(std::size_t x, std::size_t y, word_type word) {
        set_word(x / policy::X_BITS, y / policy::Y_BITS, word);
    }

    void set_word(std::size_t x, std::size_t y, word_type word) {
        words_grid[idx(x, y)] = word;
    }

    std::string debug_print_words() {
        std::ostringstream result;

        for (auto&& word : words_grid) {
            result << word << " ";
        }

        return result.str();
    }

    std::string debug_print(std::size_t line_limit = std::numeric_limits<std::size_t>::max()) const{
        std::ostringstream result;

        for (std::size_t y = 0; y < original_y_size(); ++y) {
            for (std::size_t x = 0; x < original_x_size(); ++x) {
                auto val = get_value_at(x, y) ? '1' : '0';
                result << color_0_1(val) << " ";
            }
            result << "\n";

            if (y + 1 >= line_limit) {
                return result.str();
            }
        }

        return result.str();
    }

    std::string binary_print(std::size_t line_limit = std::numeric_limits<std::size_t>::max()) const{
        std::ostringstream result;

        for (std::size_t y = 0; y < y_size(); ++y) {
            for (std::size_t x = 0; x < x_size(); ++x) {
                auto word = get_word(x, y);
                result << std::bitset<sizeof(word_type) * 8>(word) << " ";
                
            }
            result << "\n";

            if (y + 1 >= line_limit) {
                return result.str();
            }
        }

        return result.str();
    }

    size_type x_size() const {
        return _x_size;
    }

    size_type y_size() const {
        return _y_size;
    }

    size_type size() const {
        return x_size() * y_size();
    }

    size_type original_x_size() const {
        return _x_size * policy::X_BITS;
    }

    size_type original_y_size() const {
        return _y_size * policy::Y_BITS;
    }

    word_type* data() {
        return words_grid.data();
    }

    std::vector<word_type>* data_vector() {
        return &words_grid;
    }

    template <typename grid_cell_t>
    Grid<grid_cell_t> to_grid() const {
        auto _original_x_size = original_x_size();
        auto _original_y_size = original_y_size();

        Grid<grid_cell_t> grid(_original_x_size, _original_y_size);
        auto raw_data = grid.data();

        for (size_type y = 0; y < _original_y_size; y += policy::Y_BITS) {
            for (size_type x = 0; x < _original_x_size; x += policy::X_BITS) {

                auto word = get_word_from_original_coords(x, y);
                auto mask = policy::first_mask;

                for (size_type y_bit = 0; y_bit < policy::Y_BITS; ++y_bit) {
                    for (size_type x_bit = 0; x_bit < policy::X_BITS; ++x_bit) {

                        auto value = (word & mask) ? 1 : 0;

                        raw_data[in_grid_idx(x + x_bit,y + y_bit)] = static_cast<grid_cell_t>(value);
                        
                        mask = policy::move_next_mask(mask);
                    }
                }
            }
        }


        return grid;
    }

  private:
    std::size_t in_grid_idx(std::size_t x, std::size_t y) const {
        return y * original_x_size() + x;
    }

    template <typename grid_cell_t>
    void assert_dim_has_correct_size(const Grid<grid_cell_t>& grid) {
        if (grid.template size_in<1>() % policy::Y_BITS != 0) {
            throw std::invalid_argument("Grid dimensions Y must be a multiple of " + std::to_string(policy::Y_BITS));
        }
        if (grid.template size_in<0>() % policy::X_BITS != 0) {
            throw std::invalid_argument("Grid dimensions X must be a multiple of " + std::to_string(policy::X_BITS));
        }
    }

    template <typename grid_cell_t>
    void fill_grid(const Grid<grid_cell_t>& grid) {
        auto _original_x_size = original_x_size();
        auto _original_y_size = original_y_size();

        auto raw_data = grid.data();

        for (size_type y = 0; y < _original_y_size; y += policy::Y_BITS) {
            for (size_type x = 0; x < _original_x_size; x += policy::X_BITS) {

                word_type word = 0;
                auto bit_setter = policy::first_mask;
                
                for (size_type y_bit = 0; y_bit < policy::Y_BITS; ++y_bit) {
                    for (size_type x_bit = 0; x_bit < policy::X_BITS; ++x_bit) {

                        word_type value = raw_data[in_grid_idx(x + x_bit,y + y_bit)] ? 1 : 0;

                        if (value) {
                            word |= bit_setter;
                        }
                        
                        bit_setter = policy::move_next_mask(bit_setter);
                    }
                }

                set_word_from_original_coords(x, y, word);
            }
        }
    }

    std::string color_0_1(char ch) const {
        if (ch == '0') {
            return "\033[30m" + std::string(1, ch) + "\033[0m";
        }
        else {
            return "\033[31m" + std::string(1, ch) + "\033[0m";
        }
    }

    std::size_t idx(std::size_t x, std::size_t y) const {
        return y * x_size() + x;
    }

    std::vector<word_type> words_grid;
    size_type _x_size;
    size_type _y_size;
};

} // namespace algorithms

#endif // GOL_BIT_COL_GRID_HPP