#ifndef DEBUG_UTILS_PRETTY_PRINT_HPP
#define DEBUG_UTILS_PRETTY_PRINT_HPP

#include "../infrastructure/grid.hpp"
#include "../infrastructure/colors.hpp"
#include <string>
#include <vector>

namespace debug_utils {

template <typename T>
std::string pretty(const std::vector<T>& vec) {
    std::string result = "[";
    for (auto& elem : vec) {
        result += std::to_string(elem) + ", ";
    }
    result.pop_back();
    result.pop_back();
    result += "]";
    return result;
}

template <std::size_t Dims, typename ElementType>
std::string pretty_g(const infrastructure::Grid<Dims, ElementType>& grid) {
    if constexpr (Dims == 2) {   
        std::string result;

        for (std::size_t y = 0; y < grid.template size_in<1>(); ++y) {
            for (std::size_t x = 0; x < grid.template size_in<0>(); ++x) {
                std::string cell = "[";
                cell += std::to_string(static_cast<int>(grid[x][y]));
                cell += ']';

                if (grid[x][y] != 0) {
                    result += c::grid_print_one() + cell + c::reset_color();
                }
                else {
                    result += c::grid_print_zero() + cell + c::reset_color();
                }
            }
            result += "\n";
        }

        return result;
    }
    else if constexpr (Dims == 3) {
        std::string result;

        auto dimX = grid.template size_in<0>();
        auto dimY = grid.template size_in<1>();
        auto dimZ = grid.template size_in<2>();

        for (std::size_t z = 0; z < dimZ; z++) {
            result += "Layer Z=" + std::to_string(z) + "\n";
            for (std::size_t x = 0; x < dimX; ++x) {
                for (std::size_t y = 0; y < dimY; ++y) {
                    result += (grid[x][y][z] == 1 ? "O" : ".");
                }
                result += "\n";
            }
            result += "\n";
        }

        return result;
    }
    else {
        static_assert(Dims == 3 || Dims == 2, "Pretty print is only implemented for 2D and 3D grids.");
    }
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
std::string pretty(T& elem) {
    return std::to_string(elem);
}

} // namespace debug_utils

#endif // DEBUG_UTILS_PRETTY_PRINT_HPP