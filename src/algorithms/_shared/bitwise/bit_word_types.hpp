#ifndef BIT_COL_TYPES_HPP
#define BIT_COL_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace algorithms {

template <std::size_t Bits>
struct BitsConst {};

// !!! WARNING !!!
// 8 bits are not supported because it is not possible to encode the cell neighborhood in 8 bits

// template <>
// struct BitsConst<8> {
//     using word_type = std::uint8_t;
// };

template <>
struct BitsConst<16> {
    using word_type = std::uint16_t;
};

template <>
struct BitsConst<32> {
    using word_type = std::uint32_t;
};

template <>
struct BitsConst<64> {
    using word_type = std::uint64_t;
};

} // namespace algorithms

#endif // BIT_COL_TYPES_HPP