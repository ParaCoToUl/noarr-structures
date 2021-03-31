#ifndef NOARR_STRUCTURES_SCALAR_HPP
#define NOARR_STRUCTURES_SCALAR_HPP

#include "core.hpp"
#include "struct_decls.hpp"
#include "contain.hpp"

namespace noarr {

template<typename T, typename... KS>
struct _scalar_get_t;

template<typename T>
struct _scalar_get_t<T> {
    using type = T;
};

template<typename T>
struct _scalar_get_t<T, void> {
    using type = T;
};

/**
 * @brief The ground structure
 * 
 * @tparam T the stored type
 */
template<typename T>
struct scalar {
    static constexpr std::tuple<> sub_structures() { return {}; }
    using description = struct_description<
        char_pack<'s', 'c', 'a', 'l', 'a', 'r'>,
        dims_impl<>,
        dims_impl<>,
        type_param<T>>;

    template<typename... KS>
    using get_t = typename _scalar_get_t<T, KS...>::type;

    constexpr scalar() = default;
    static constexpr auto construct() {
        return scalar<T>();
    }
    static constexpr std::size_t size() { return sizeof(T); }
    static constexpr std::size_t offset() { return 0; }
    static constexpr std::size_t length() { return 0; }
};

}

#endif // NOARR_STRUCTURES_SCALAR_HPP
