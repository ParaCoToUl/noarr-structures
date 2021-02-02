#ifndef NOARR_FUNCS_HPP
#define NOARR_FUNCS_HPP

#include "noarr_structs.hpp"

namespace noarr {

template<char DIM>
struct resize {
    const std::size_t length;

    explicit constexpr resize(std::size_t length) : length{length} {}

    template<typename T>
    constexpr auto operator()(vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures), length};
    }
};

template<char DIM, std::size_t L>
struct cresize {
    constexpr cresize() {}

    template<typename T>
    constexpr auto operator()(vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
    template<typename T>
    constexpr auto operator()(array<DIM, L, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures)};
    }
};

template<typename T, std::size_t i, typename = void>
struct safe_get_ {
    static constexpr void get(T) {

    }
};

template<typename T, std::size_t i>
struct safe_get_<T, i, std::enable_if_t<(std::tuple_size<remove_cvref<decltype(T::sub_structures)>>::value > i)>> {
    static constexpr auto get(T t) {
        return std::get<i>(t.sub_structures);
    }
};

template<std::size_t i, typename T>
inline constexpr auto safe_get(T t) {
    return safe_get_<T, i>::get(t);
}

template<char DIM>
struct fix {
    const std::size_t idx;

    explicit constexpr fix(std::size_t idx) : idx{idx} {}

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<dims_have<T, DIM>::value>>(), fixed_dim<decltype(safe_get<0>(t))>{safe_get<0>(t), idx}) {
        return fixed_dim<decltype(safe_get<0>(t))>{safe_get<0>(t), idx};
    }
};

template<char... DIMS>
struct fixs;

template<char DIM, char... DIMS>
struct fixs<DIM, DIMS...> {
    const std::size_t idx_;
    const fixs<DIMS...> fixs_;

    template <typename... IDXs>
    constexpr fixs(std::size_t idx, IDXs... idxs) : idx_{idx}, fixs_{idxs...} {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, fix<DIM>{idx_}, fixs_);
    }
};

template<char DIM>
struct fixs<DIM> : fix<DIM> {
    explicit constexpr fixs(std::size_t idx) : fix<DIM>{idx} {}
};

template<char DIM>
struct get_offset {
    const std::size_t idx;
    using func_family = get_trait;

    explicit constexpr get_offset(std::size_t idx) : idx{idx} {}

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<dims_have<T, DIM>::value>>(), t.offset(idx)) {
        return t.offset(idx);
    }
};

// TODO: implement recursive offset
struct offset {
    using func_family = top_trait;
    explicit constexpr offset() {}

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(t.offset()) {
        return t.offset();
    }
};

struct get_size {
    using func_family = top_trait;
    constexpr get_size() {}

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(t.size()) {
        return t.size();
    }
};

}

#endif // NOARR_FUNCS_HPP