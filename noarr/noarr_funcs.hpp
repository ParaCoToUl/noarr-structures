#ifndef NOARR_FUNCS_HPP
#define NOARR_FUNCS_HPP

#include "noarr_structs.hpp"

namespace noarr {

template<char Dim>
struct resize {
    using func_family = transform_tag;
    explicit constexpr resize(std::size_t length) : length{length} {}

    template<typename T>
    constexpr auto operator()(vector<Dim, T> v) const {
        return sized_vector<Dim, T>{std::get<0>(v.sub_structures()), length};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<Dim, T> v) const {
        return sized_vector<Dim, T>{std::get<0>(v.sub_structures()), length};
    }

private:
    std::size_t length;
};

template<char Dim>
struct _reassemble_get {
    using func_family = get_tag;

    template<typename T>
    constexpr auto operator()(T t) const -> remove_cvref<decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>()>>(), t)> {
        return t;
    }
};

template<char Dim, typename T, typename T2>
struct _reassemble_set : contain<T> {
    constexpr _reassemble_set() = default;
    explicit constexpr _reassemble_set(T t) : contain<T>{t} {}
    using func_family = transform_tag;

    constexpr auto operator()(T2 t) const {
        return construct(contain<T>::template get<0>(), t.sub_structures());
    }
};

template<char Dim1, char Dim2>
struct reassemble {
private:
    template<char Dim, typename T, typename T2>
    constexpr auto reassemble_2(T t, T2 t2) const {
        return construct(t2, (t % _reassemble_set<Dim, T, remove_cvref<decltype(t2)>>{t}).sub_structures());
    }
    template<char Dim, typename T>
    constexpr auto reassemble_(T t) const -> decltype(reassemble_2<Dim>(t, t % _reassemble_get<Dim>{})) {
        return reassemble_2<Dim>(t, t % _reassemble_get<Dim>{});
    }
public:
    using func_family = transform_tag;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(reassemble_<std::enable_if_t<get_dims<T>::template contains<Dim1>(), char>(Dim2)>(t)) {
        return reassemble_<Dim2>(t);
    }

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(reassemble_<std::enable_if_t<get_dims<T>::template contains<Dim2>() && Dim1 != Dim2, char>(Dim1)>(t)) {
        return reassemble_<Dim1>(t);
    }
};

template<char Dim, std::size_t L>
struct cresize {
    constexpr cresize() = default;

    template<typename T>
    constexpr auto operator()(vector<Dim, T> v) const {
        return array<Dim, L, T>{std::get<0>(v.sub_structures())};
    }
    template<typename T>
    constexpr auto operator()(sized_vector<Dim, T> v) const {
        return array<Dim, L, T>{std::get<0>(v.sub_structures())};
    }
    template<typename T>
    constexpr auto operator()(array<Dim, L, T> v) const {
        return array<Dim, L, T>{std::get<0>(v.sub_structures())};
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

// TODO?: implement cfix and cfixs
// TODO: support fix and fixs somehow on tuples
// TODO: support the arrr::at functor

template<char Dim>
struct fix {
    explicit constexpr fix(std::size_t idx) : idx{idx} {}

private:
    std::size_t idx;

public:
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>()>>(), fixed_dim<Dim, T>{t, idx}) {
        return fixed_dim<Dim, T>{t, idx};
    }
};

template<char... Dims>
struct fixs;

template<char Dim, char... Dims>
struct fixs<Dim, Dims...> : private fixs<Dims...> {
    template <typename... IS>
    constexpr fixs(std::size_t idx, IS... is) : fixs<Dims...>{static_cast<size_t>(is)...}, idx_{idx} {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, fix<Dim>{idx_}, static_cast<const fixs<Dims...>&>(*this));
    }

private:
    std::size_t idx_;
};

template<char Dim>
struct fixs<Dim> : fix<Dim> {
    explicit constexpr fixs(std::size_t idx) : fix<Dim>{idx} {}
};

template<char Dim>
struct get_offset {
    using func_family = get_tag;
    explicit constexpr get_offset(std::size_t idx) : idx{idx} {}

private:
    std::size_t idx;

public:
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>()>>(), t.offset(idx)) {
        return t.offset(idx);
    }
};

struct offset {
    using func_family = top_tag;
    explicit constexpr offset() = default;

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
    using func_family = top_tag;
    constexpr get_size() = default;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(t.size()) {
        return t.size();
    }
};

}

#endif // NOARR_FUNCS_HPP
