#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "structs.hpp"
#include "struct_traits.hpp"
#include "core.hpp"

// TODO: rename resize to set_length
// TODO: rename fixs to fix ---> maybe even sfixs to fix as it is the most general one and supplements all others
// TODO: rework get_at so it works as fix... | get_at on non-points <--/
//                                                                 (this change would apply to this TODO as well)

namespace noarr {

template<char Dim>
struct resize {
    using func_family = transform_tag;
    explicit constexpr resize(std::size_t length) : length(length) {}

    template<typename T>
    constexpr auto operator()(vector<Dim, T> v) const {
        return sized_vector<Dim, T>(std::get<0>(v.sub_structures()), length);
    }
    template<typename T>
    constexpr auto operator()(sized_vector<Dim, T> v) const {
        return sized_vector<Dim, T>(std::get<0>(v.sub_structures()), length);
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
struct _reassemble_set : private contain<T> {
    constexpr _reassemble_set() = default;
    explicit constexpr _reassemble_set(T t) : contain<T>(t) {}
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
        return construct(t2, (t | _reassemble_set<Dim, T, remove_cvref<decltype(t2)>>(t)).sub_structures());
    }
    template<char Dim, typename T>
    constexpr auto reassemble_(T t) const -> decltype(reassemble_2<Dim>(t, t | _reassemble_get<Dim>())) {
        return reassemble_2<Dim>(t, t | _reassemble_get<Dim>());
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
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
    }
    template<typename T>
    constexpr auto operator()(sized_vector<Dim, T> v) const {
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
    }
    template<typename T>
    constexpr auto operator()(array<Dim, L, T> v) const {
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
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

template<char Dim>
struct fix {
    constexpr fix() = default;
    explicit constexpr fix(std::size_t idx) : idx(idx) {}

private:
    std::size_t idx;

public:
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>()>>(), fixed_dim<Dim, T>(t, idx)) {
        return fixed_dim<Dim, T>(t, idx);
    }
};

template<char... Dims>
struct fixs;

template<char Dim, char... Dims>
struct fixs<Dim, Dims...> : private contain<fix<Dim>, fixs<Dims...>> {
private:
    using base = contain<fix<Dim>, fixs<Dims...>>;
public:
    constexpr fixs() = default;
    template <typename... IS>
    constexpr fixs(std::size_t idx, IS... is) : base(fix<Dim>(idx), fixs<Dims...>(static_cast<std::size_t>(is)...)) {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, base::template get<0>(), base::template get<1>());
    }
};

template<char Dim>
struct fixs<Dim> : fix<Dim> { using fix<Dim>::fix; using fix<Dim>::operator(); };

template<char Dim, std::size_t Idx>
struct _sfix {
    using idx_t = std::integral_constant<std::size_t, Idx>;
    constexpr _sfix() = default;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>()>>(), sfixed_dim<Dim, T, Idx>(t)) {
        return sfixed_dim<Dim, T, Idx>(t);
    }
};

template<char Dim, std::size_t Idx>
inline constexpr auto sfix(std::integral_constant<std::size_t, Idx>) {
    return _sfix<Dim, Idx>();
}

template<typename... Tuples>
struct _sfixs;

template<char Dim, typename T, typename... Tuples>
struct _sfixs<std::tuple<std::integral_constant<char, Dim>, T>, Tuples...> : private contain<fix<Dim>, _sfixs<Tuples...>> {
    using base = contain<fix<Dim>, _sfixs<Tuples...>>;
    constexpr _sfixs() = default;
    
    template <typename... Ts>
    constexpr _sfixs(T t, Ts... ts) : base(fix<Dim>(t), _sfixs<Tuples...>(ts...)) {}

    template<typename S>
    constexpr auto operator()(S s) const {
        return pipe(s, base::template get<0>(), base::template get<1>());
    }
};

template<char Dim, typename T>
struct _sfixs<std::tuple<std::integral_constant<char, Dim>, T>> : fix<Dim> { using fix<Dim>::fix; using fix<Dim>::operator(); };


template<char Dim, std::size_t Idx, typename... Tuples>
struct _sfixs<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>, Tuples...> : contain<_sfix<Dim, Idx>, _sfixs<Tuples...>> {
    using base = contain<fix<Dim>, _sfixs<Tuples...>>;
    constexpr _sfixs() = default;
    
    template <typename... Ts>
    constexpr _sfixs(Ts... ts) : base(_sfix<Dim, Idx>(), _sfixs<Tuples...>(ts...)) {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, base::template get<0>(), base::template get<1>());
    }
};

template<char Dim, std::size_t Idx>
struct _sfixs<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>> : _sfix<Dim, Idx> {
    constexpr _sfixs() = default;
    constexpr _sfixs(std::integral_constant<std::size_t, Idx>) : _sfix<Dim,Idx>() {}
    using _sfix<Dim, Idx>::operator();
};

template<char... Dims, typename... Ts>
inline constexpr auto sfixs(Ts... ts) {
    return _sfixs<std::tuple<std::integral_constant<char, Dims>, Ts>...>(ts...);
}

template<char Dim>
struct get_offset {
    using func_family = get_tag;
    explicit constexpr get_offset(std::size_t idx) : idx(idx) {}

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
    constexpr std::size_t operator()(scalar<T>) {
        return 0;
    }

    template<typename T>
    constexpr auto operator()(T t) const -> std::enable_if_t<is_point<T>::value, std::size_t> {
        return t.offset() + (std::get<0>(t.sub_structures()) | offset());
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

struct get_at : private contain<char*> {
    using func_family = top_tag;

    constexpr get_at() = delete;

    template<typename T>
    explicit constexpr get_at(T *ptr) : contain<char *>(reinterpret_cast<char *>(ptr)) {}


    template<typename T>
    constexpr auto operator()(T t) const -> std::enable_if_t<is_cube<T>::value, scalar_t<T> &> {
        return reinterpret_cast<scalar_t<T> &>(*(contain<char *>::template get<0>() + (t | offset())));
    }
};

}

#endif // NOARR_STRUCTURES_FUNCS_HPP
