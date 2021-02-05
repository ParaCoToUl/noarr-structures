#ifndef NOARR_FUNCS_HPP
#define NOARR_FUNCS_HPP

#include "noarr_structs.hpp"

namespace noarr {

template<char DIM>
struct resize {
    const std::size_t length;
    using func_family = transform_trait;

    explicit constexpr resize(std::size_t length) : length{length} {}

    template<typename T>
    constexpr auto operator()(vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures()), length};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<DIM, T> v) const {
        return sized_vector<DIM, T>{std::get<0>(v.sub_structures()), length};
    }
};

template<char DIM, std::size_t L>
struct cresize {
    constexpr cresize() {}

    template<typename T>
    constexpr auto operator()(vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures())};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<DIM, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures())};
    }
    template<typename T>
    constexpr auto operator()(array<DIM, L, T> v) const {
        return array<DIM, L, T>{std::get<0>(v.sub_structures())};
    }
};

template<typename T, std::size_t i, typename = void>
struct safe_get_ {
    static constexpr void get(T t) = delete;
};

template<typename T, std::size_t i>
struct safe_get_<T, i, std::enable_if_t<(std::tuple_size<remove_cvref<decltype(std::declval<T>().sub_structures())>>::value > i)>> {
    static constexpr auto get(T t) {
        return std::get<i>(t.sub_structures());
    }
};

template<std::size_t i, typename T>
inline constexpr auto safe_get(T t) {
    return safe_get_<T, i>::get(t);
}

// TODO?: cfix & cfixs
// TODO: support fix and fixs somehow on tuples
// TODO: support the arrr::at functor

template<char DIM>
struct fix {
    const std::size_t idx;

    explicit constexpr fix(std::size_t idx) : idx{idx} {}

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<dims_have<T, DIM>::value>>(), fixed_dim<DIM, T>{t, idx}) {
        return fixed_dim<DIM, T>{t, idx};
    }
};

template<char... DIMS>
struct fixs;

template<char DIM, char... DIMS>
struct fixs<DIM, DIMS...> : private fixs<DIMS...> {
    const std::size_t idx_;

    template <typename... IDXs>
    constexpr fixs(std::size_t idx, IDXs... idxs) : idx_{idx}, fixs<DIMS...>{static_cast<size_t>(idxs)...} {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, fix<DIM>{idx_}, static_cast<const fixs<DIMS...>&>(*this));
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

struct offset {
    using func_family = top_trait;
    explicit constexpr offset() {}

    template<typename T>
    constexpr std::size_t operator()(scalar<T>) const {
        return 0;
    }

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<typename T::template get_t<>>(), t.offset()) {
        return t.offset() + (std::get<0>(t.sub_structures()) % offset{});
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
