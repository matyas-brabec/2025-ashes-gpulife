#ifndef DIFF_GRID_HPP
#define DIFF_GRID_HPP

#include <cstddef>
#include <sstream>
#include <string>

#include "../infrastructure/grid.hpp"
#include "./pretty_print.hpp"

namespace debug_utils {

template <int Dims, typename ElementType>
std::string diff(const infrastructure::Grid<Dims, ElementType>& original,
                        const infrastructure::Grid<Dims, ElementType>& other) {

    std::ostringstream diff_str;

    auto original_data = original.data();
    auto other_data = other.data();

    for (size_t i = 0; i < original.size(); i++) {
        if (original_data[i] != other_data[i]) {
            auto coords = original.idx_to_coordinates(i);

            diff_str << "at: " << i << " ~ " << pretty(coords) << ": " << std::to_string(original_data[i])
                     << " != \033[31m" << std::to_string(other_data[i]) << "\033[0m" << std::endl;
        }
    }
    return diff_str.str();
}

template <typename ElementType>
std::string diff(const infrastructure::Grid<2, ElementType>& original,
                        const infrastructure::Grid<2, ElementType>& other) {

    std::ostringstream diff_str;

    auto x_size = original.size_in(0);
    auto y_size = original.size_in(1);

    const std::size_t printed_tiles_size = 8;

    for (std::size_t y = 0; y < y_size; y++) {
        for (std::size_t x = 0; x < x_size; x++) {

            if ((x % printed_tiles_size == 0) && (x != 0)) {
                diff_str << " ";
            }

            if (x != 0) {
                diff_str << " ";
            }

            if (original[x][y] != other[x][y]) {
                // diff_str << "\033[31m" << std::to_string(other[x][y]) << "\033[34m" << std::to_string(original[x][y])
                // << "\033[0m";
                diff_str << "\033[31m" << std::to_string(other[x][y]) << "\033[0m";
            }
            else {
                auto color = original[x][y] == 0 ? "\033[30m" : "\033[33m";
                diff_str << color << std::to_string(original[x][y]) << "\033[0m";
            }
        }

        if ((y + 1) % printed_tiles_size == 0) {
            diff_str << "\n";
        }

        diff_str << "\n";
    }

    return diff_str.str();
}

}; // namespace debug_utils

#endif // DIFF_GRID_HPP
