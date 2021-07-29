#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "struct_traits.hpp"
#include "core.hpp"

namespace noarr {

// TODO: function that returns the topmost dimension

namespace literals {

template<std::size_t Accum, char... Chars>
struct _idx_translate;

template<std::size_t Accum, char Char, char... Chars>
struct _idx_translate<Accum, Char, Chars...> {
    using type = typename _idx_translate<Accum * 10 + (std::size_t)(Char - '0'), Chars...>::type;
};

template<std::size_t Accum, char Char>
struct _idx_translate<Accum, Char> {
    using type = std::integral_constant<std::size_t, Accum * 10 + (std::size_t)(Char - '0')>;
};

/**
 * @brief Converts an integer literal into a corresponding std::integral_constant<std::size_t, ...>
 * 
 * @tparam Chars the digits of the integer literal
 * @return constexpr auto the corresponding std::integral_constant
 */
template<char... Chars>
inline constexpr auto operator""_idx() {
    return typename _idx_translate<0, Chars...>::type();
}

}

template<typename F, typename G>
struct _compose : contain<F, G> {
    using base = contain<F, G>;
    using func_family = typename func_trait<F>::type;

    template<typename T>
    using can_apply = get_applicability<F, T>;

    constexpr _compose(F f, G g) : base(f, g) {}

    template<typename T>
    constexpr decltype(auto) operator()(T t) const {
        return pipe(t, base::template get<0>(), base::template get<1>());
    }
};

/**
 * @brief composes functions `F` and `G` together
 * 
 * @param f: the inner function
 * @param g: the outer function
 */
template<typename F, typename G>
inline constexpr decltype(auto) compose(F f, G g) {
    return _compose<F, G>(f, g);
}

template<char Dim, typename T>
struct _set_length_can_apply {
    static constexpr bool value = false;
};

template<char Dim, typename T>
struct _set_length_can_apply<Dim, vector<Dim, T>> {
    static constexpr bool value = true;
};

template<char Dim, typename T>
struct _set_length_can_apply<Dim, sized_vector<Dim, T>> {
    static constexpr bool value = true;
};

template<char Dim>
struct _set_length {
    using func_family = transform_tag;

    template<typename T>
    using can_apply = _set_length_can_apply<Dim, T>;

    explicit constexpr _set_length(std::size_t length) : length(length) {}

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

template<char Dim, std::size_t L>
struct _sset_length {
    constexpr _sset_length() = default;

    template<typename T>
    constexpr auto operator()(vector<Dim, T> v) const {
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
    }

    template<typename T>
    constexpr auto operator()(sized_vector<Dim, T> v) const {
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
    }

    template<typename T, std::size_t L2>
    constexpr auto operator()(array<Dim, L2, T> v) const {
        return array<Dim, L, T>(std::get<0>(v.sub_structures()));
    }
};

/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim>
inline constexpr auto set_length(std::size_t length) {
    return _set_length<Dim>(length);
}

/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim, std::size_t Length>
inline constexpr auto set_length(std::integral_constant<std::size_t, Length>) {
    return _sset_length<Dim, Length>();
}

/**
 * @brief returns the number of indices in the structure specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the desired structure
 */
template<char Dim>
struct get_length {
    using func_family = get_tag;

    template<typename T>
    using can_apply = typename get_dims<T>::template contains<Dim>;

    explicit constexpr get_length() {}

    template<typename T>
    constexpr std::size_t operator()(T t) const {
        return t.length();
    }
};

template<char Dim>
struct _reassemble_get {
    using func_family = get_tag;

    template<typename T>
    constexpr auto operator()(T t) const -> remove_cvref<decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), t)> {
        return t;
    }
};

template<char Dim, typename T, typename T2>
struct _reassemble_set : private contain<T> {
    using func_family = transform_tag;

    constexpr _reassemble_set() = default;
    explicit constexpr _reassemble_set(T t) : contain<T>(t) {}

    constexpr auto operator()(T2 t) const {
        return construct(contain<T>::template get<0>(), t.sub_structures());
    }
};

/**
 * @brief swaps two structures given by their dimension names in the substructure tree of a structure
 * 
 * @tparam Dim1: the dimension name of the first structure
 * @tparam Dim2: the dimension name of the second structure
 */
template<char Dim1, char Dim2>
struct reassemble {
private:
    template<char Dim, typename T, typename T2>
    constexpr auto reassemble_2(T t, T2 t2) const {
        return construct(t2, (t | _reassemble_set<Dim, T, remove_cvref<decltype(t2)>>(t)).sub_structures());
    }

    template<char Dim, typename T>
    constexpr auto _reassemble(T t) const -> decltype(reassemble_2<Dim>(t, t | _reassemble_get<Dim>())) {
        return reassemble_2<Dim>(t, t | _reassemble_get<Dim>());
    }
public:
    using func_family = transform_tag;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(_reassemble<std::enable_if_t<get_dims<T>::template contains<Dim1>::value, char>(Dim2)>(t)) {
        return _reassemble<Dim2>(t);
    }

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(_reassemble<std::enable_if_t<get_dims<T>::template contains<Dim2>::value && Dim1 != Dim2, char>(Dim1)>(t)) {
        return _reassemble<Dim1>(t);
    }
};

template<typename T, std::size_t i, typename = void>
struct _safe_get {
    static constexpr void get(T t) = delete;
};

template<typename T, std::size_t i>
struct _safe_get<T, i, std::enable_if_t<(std::tuple_size<remove_cvref<decltype(std::declval<T>().sub_structures())>>::value > i)>> {
    static constexpr auto get(T t) {
        return std::get<i>(t.sub_structures());
    }
};

template<std::size_t i, typename T>
inline constexpr auto safe_get(T t) {
    return _safe_get<T, i>::get(t);
}

template<char Dim>
struct _fix {
    constexpr _fix() = default;
    explicit constexpr _fix(std::size_t idx) : idx(idx) {}

private:
    std::size_t idx;

public:
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), fixed_dim<Dim, T>(t, idx)) {
        return fixed_dim<Dim, T>(t, idx);
    }
};

template<char Dim, std::size_t Idx>
struct _sfix {
    using idx_t = std::integral_constant<std::size_t, Idx>;

    constexpr _sfix() = default;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), sfixed_dim<Dim, T, Idx>(t)) {
        return sfixed_dim<Dim, T, Idx>(t);
    }
};

template<typename... Tuples>
struct _fixs;

template<char Dim, typename T, typename... Tuples>
struct _fixs<std::tuple<std::integral_constant<char, Dim>, T>, Tuples...> : private contain<_fix<Dim>, _fixs<Tuples...>> {
    using base = contain<_fix<Dim>, _fixs<Tuples...>>;

    constexpr _fixs() = default;
    
    template <typename... Ts>
    constexpr _fixs(T t, Ts... ts) : base(_fix<Dim>(t), _fixs<Tuples...>(ts...)) {}

    template<typename S>
    constexpr auto operator()(S s) const {
        return pipe(s, base::template get<0>(), base::template get<1>());
    }
};

template<char Dim, typename T>
struct _fixs<std::tuple<std::integral_constant<char, Dim>, T>> : private _fix<Dim> { using _fix<Dim>::_fix; using _fix<Dim>::operator(); };


template<char Dim, std::size_t Idx, typename... Tuples>
struct _fixs<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>, Tuples...> : private contain<_sfix<Dim, Idx>, _fixs<Tuples...>> {
    using base = contain<_sfix<Dim, Idx>, _fixs<Tuples...>>;

    constexpr _fixs() = default;
    
    template <typename... Ts>
    constexpr _fixs(std::integral_constant<std::size_t, Idx>, Ts... ts) : base(_sfix<Dim, Idx>(), _fixs<Tuples...>(ts...)) {}

    template<typename T>
    constexpr auto operator()(T t) const {
        return pipe(t, base::template get<0>(), base::template get<1>());
    }
};

template<char Dim, std::size_t Idx>
struct _fixs<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>> : private _sfix<Dim, Idx> {
    constexpr _fixs() = default;
    constexpr _fixs(std::integral_constant<std::size_t, Idx>) : _sfix<Dim,Idx>() {}

    using _sfix<Dim, Idx>::operator();
};

template<>
struct _fixs<> {
    constexpr _fixs() = default;

    template<typename T>
    constexpr auto operator()(T t) const {
        return t;
    }
};

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, typename... Ts>
inline constexpr auto fix(Ts... ts) {
    return _fixs<std::tuple<std::integral_constant<char, Dims>, Ts>...>(ts...);
}

/**
 * @brief returns the offset of a substructure given by a dimension name in a structure
 * 
 * @tparam Dim: the dimension name
 */
template<char Dim>
struct get_offset {
    using func_family = get_tag;

    explicit constexpr get_offset(std::size_t idx) : idx(idx) {}

private:
    std::size_t idx;

public:
    template<typename T>
    constexpr auto operator()(T t) const -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), t.offset(idx)) {
        return t.offset(idx);
    }
};

struct _offset {
    using func_family = top_tag;
    explicit constexpr _offset() = default;

    template<typename T>
    constexpr std::size_t operator()(scalar<T>) {
        return 0;
    }

    template<typename T>
    constexpr auto operator()(T t) const -> std::enable_if_t<is_point<T>::value, std::size_t> {
        return t.offset() + (std::get<0>(t.sub_structures()) | _offset());
    }
};

inline constexpr auto offset() {
    return _offset();
}

/**
 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item 
 * 
 * @tparam Dims: the dimension names of fixed indices
 * @param ts: parameters for fixing the indices
 * @return constexpr auto 
 */
template<char... Dims, typename... Ts>
inline constexpr auto offset(Ts... ts) {
    return compose(fix<Dims...>(ts...), _offset());
}

/**
 * @brief returns the size (in bytes) of the structure
 * 
 */
struct get_size {
    using func_family = top_tag;
    constexpr get_size() = default;

    template<typename T>
    constexpr auto operator()(T t) const -> decltype(t.size()) {
        return t.size();
    }
};

struct _get_at : private contain<char*> {
    using func_family = top_tag;

    constexpr _get_at() = delete;

    template<typename T>
    explicit constexpr _get_at(T *ptr) : contain<char *>(reinterpret_cast<char *>(ptr)) {}


    template<typename T>
    constexpr auto operator()(T t) const -> std::enable_if_t<is_cube<T>::value, scalar_t<T> &> {
        return reinterpret_cast<scalar_t<T> &>(*(contain<char *>::template get<0>() + (t | offset())));
    }
};

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 * 
 * @param ptr: the pointer to blob structure
 */
template<typename V>
inline constexpr decltype(auto) get_at(V *ptr) {
    return _get_at(ptr);
}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, typename V, typename... Ts>
inline constexpr decltype(auto) get_at(V *ptr, Ts... ts) {
    return compose(fix<Dims...>(ts...), _get_at(ptr));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_FUNCS_HPP
