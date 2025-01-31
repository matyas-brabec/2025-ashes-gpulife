#ifndef ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP
#define ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP

#include <bitset>
#include <cstddef>
#include <iostream>
#include <sstream>

#include "../../template_helpers/static_for.hpp"
#include "../bit_modes.hpp"
namespace algorithms {

enum class Position {
    TOP = 0,
    BOTTOM = 1,
};

template <Position POSITION, typename word_type>
class MasksByPosition {};

template <typename word_type>
struct BitwiseColsTemplatedOps {
    constexpr static std::size_t BITS_IN_COL = sizeof(word_type) * 8;

    template <Position POSITION>
    using masks = MasksByPosition<POSITION, word_type>;

    using bit_grid_mode = BitColumnsMode;

    // clang-format off
    static word_type compute_center_word(
        word_type lt, word_type ct, word_type rt, 
        word_type lc, word_type cc, word_type rc,
        word_type lb, word_type cb, word_type rb) {
        
        word_type result = compute_inner_bits(lc, cc, rc);

        result |= compute_side_col<Position::TOP>(
            lc, cc, rc,
            lt, ct, rt);

        result |= compute_side_col<Position::BOTTOM>(
            lc, cc, rc,
            lb, cb, rb);

        return result; 
    }
    // clang-format on

  private:
    static word_type compute_inner_bits(word_type lc, word_type cc, word_type rc) {
        word_type result = 0;

        templates::static_for<1, BITS_IN_COL - 1>::run(
            [&lc, &cc, &rc, &result]<std::size_t N>() { result |= compute_inner_cell<N>(lc, cc, rc); });

        return result;
    }

    template <std::size_t N>
    static word_type compute_inner_cell(word_type lc, word_type cc, word_type rc) {
        word_type result = 0;
        constexpr word_type cell_mask = static_cast<word_type>(0b010) << (N - 1);
        constexpr word_type one = cell_mask;

        auto cell = cc & cell_mask;

        auto neighborhood = combine_neighborhoods_into_one_word<N>(lc, cc, rc);
        auto alive_neighbours = __builtin_popcountll(neighborhood);

        // auto alive_neighbours =
        //     __builtin_popcountll(lc & cell_mask) +
        //     __builtin_popcountll(cc & cell_mask) +
        //     __builtin_popcountll(rc & cell_mask);

        if (cell != 0) {
            if (alive_neighbours < 2 || alive_neighbours > 3) {
                result &= ~one;
            }
            else {
                result |= one;
            }
        }
        else {
            if (alive_neighbours == 3) {
                result |= one;
            }
            else {
                result &= ~one;
            }
        }

        return result;
    }

    template <std::size_t N>
    static word_type combine_neighborhoods_into_one_word(word_type lc, word_type cc, word_type rc) {

        constexpr word_type site_neighborhood_mask = static_cast<word_type>(0b111) << (N - 1);
        constexpr word_type center_neighborhood_mask = static_cast<word_type>(0b101) << (N - 1);
        constexpr word_type NEIGHBORHOOD_WINDOW = 6;

        return offset<6, N - 1, NEIGHBORHOOD_WINDOW>(lc & site_neighborhood_mask) |
               offset<3, N - 1, NEIGHBORHOOD_WINDOW>(cc & center_neighborhood_mask) | (rc & site_neighborhood_mask);
    }

    template <std::size_t N, std::size_t CENTER, std::size_t NEIGHBORHOOD_WINDOW>
    static word_type offset(word_type num) {
        if constexpr (CENTER < NEIGHBORHOOD_WINDOW) {
            return num << N;
        }
        else {
            return num >> N;
        }
    }

    // clang-format off
    template <Position POSITION>
    static word_type compute_side_col(
        word_type cl, word_type cc, word_type cr,
        word_type l_, word_type c_, word_type r_) {

        constexpr word_type SITE_MASK = masks<POSITION>::SITE_MASK;
        constexpr word_type CENTER_MASK = masks<POSITION>::CENTER_MASK;
        constexpr word_type UP_BOTTOM_MASK = masks<POSITION>::UP_BOTTOM_MASK;
        constexpr word_type CELL_MASK = masks<POSITION>::CELL_MASK;

        auto neighborhood = 
            masks<POSITION>::template offset_center_cols<7>(cl & SITE_MASK) | 
            masks<POSITION>::template offset_center_cols<5>(cc & CENTER_MASK) |
            masks<POSITION>::template offset_center_cols<3>(cr & SITE_MASK) |
            masks<POSITION>::template offset_top_bottom_cols<2>(l_ & UP_BOTTOM_MASK) |
            masks<POSITION>::template offset_top_bottom_cols<1>(c_ & UP_BOTTOM_MASK) |
                                                               (r_ & UP_BOTTOM_MASK);
        
        auto cell = cc & CELL_MASK;

        // auto alive_neighbours = 
        //     __builtin_popcountll(cl & SITE_MASK) +
        //     __builtin_popcountll(cc & CENTER_MASK) +
        //     __builtin_popcountll(cr & SITE_MASK) +
        //     __builtin_popcountll(l_ & UP_BOTTOM_MASK) +
        //     __builtin_popcountll(c_ & UP_BOTTOM_MASK) +
        //     __builtin_popcountll(r_ & UP_BOTTOM_MASK);

        auto alive_neighbours = __builtin_popcountll(neighborhood);

        if (cell != 0) {
            if (alive_neighbours < 2 || alive_neighbours > 3) {
                return 0;
            }
            else {
                return CELL_MASK;
            }
        }
        else {
            if (alive_neighbours == 3) {
                return CELL_MASK;
            }
            else {
                return 0;
            }
        }

    }
    // clang-format on
};

template <typename word_type>
class MasksByPosition<Position::TOP, word_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(word_type) * 8;

    static constexpr word_type SITE_MASK = 0b11;
    static constexpr word_type CENTER_MASK = 0b10;
    static constexpr word_type UP_BOTTOM_MASK = static_cast<word_type>(1) << (BITS_IN_COL - 1);
    static constexpr word_type CELL_MASK = 0b1;

    template <std::size_t N>
    static word_type offset_center_cols(word_type num) {
        return num << N;
    }

    template <std::size_t N>
    static word_type offset_top_bottom_cols(word_type num) {
        return num >> N;
    }
};

template <typename word_type>
class MasksByPosition<Position::BOTTOM, word_type> {
  public:
    static constexpr std::size_t BITS_IN_COL = sizeof(word_type) * 8;

    static constexpr word_type SITE_MASK = static_cast<word_type>(0b11) << (BITS_IN_COL - 2);
    static constexpr word_type CENTER_MASK = static_cast<word_type>(0b01) << (BITS_IN_COL - 2);
    static constexpr word_type UP_BOTTOM_MASK = 1;
    static constexpr word_type CELL_MASK = static_cast<word_type>(0b1) << (BITS_IN_COL - 1);

    template <std::size_t N>
    static word_type offset_center_cols(word_type num) {
        return num >> N;
    }

    template <std::size_t N>
    static word_type offset_top_bottom_cols(word_type num) {
        return num << N;
    }
};

} // namespace algorithms

#endif // ALGORITHMS_SHARED_BITWISE_GOL_OPERATION_HPP