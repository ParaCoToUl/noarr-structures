#ifndef NOARR_FUNCS_HPP
#define NOARR_FUNCS_HPP

#include "noarr_structs.hpp"

namespace noarr {

template<char DIM>
struct resize {
    const std::size_t length;
    explicit constexpr resize(std::size_t length) : length{length} {}
    template<typename T>
    auto operator()(vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
    template<typename T>
    auto operator()(sized_vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
};

template<char DIM, std::size_t L>
struct cresize {
    constexpr cresize() {}
    template<typename T>
    auto operator()(vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    auto operator()(sized_vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    constexpr auto operator()(array<DIM, L, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
};

template<char DIM>
struct get_offset {
    const std::size_t idx;
    explicit constexpr get_offset(std::size_t idx) : idx{idx} {}
    using func_family = get_trait;
    
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<!dims_have<T, DIM>::value>>(), t.offset(idx)) {
        return t.offset(idx);
    }
};

struct get_size {
    constexpr get_size() {}
    using func_family = get_trait;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(t.size()) {
        return t.size();
    }
};

}

#endif // NOARR_FUNCS_HPP