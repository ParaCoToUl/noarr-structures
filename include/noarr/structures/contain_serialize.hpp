#ifndef NOARR_STRUCTURES_CONTAIN_SERIALIZE_HPP
#define NOARR_STRUCTURES_CONTAIN_SERIALIZE_HPP

#include <cstring>

#include "contain.hpp"

namespace noarr {

/**
 * @brief serializes a structure to an array of bytes
 * 
 * @param contain: the structure (contain is it's base class)
 * @param bytes: the array of bytes
 */
template<class... Types>
inline void serialize(const contain<Types...> &contain, char *bytes) {
    std::memcpy(bytes, &contain, sizeof(contain));
} // namespace noarr

/**
 * @brief deserializes a structure from an array of bytes
 * 
 * @param contain: the structure (contain is it's base class)
 * @param bytes: the array of bytes
 */
template<class... Types>
inline void deserialize(contain<Types...> &contain, const char *bytes) {
    std::memcpy(&contain, bytes, sizeof(contain));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CONTAIN_SERIALIZE_HPP