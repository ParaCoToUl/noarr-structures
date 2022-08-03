#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "struct_getters.hpp"
#include "pipes.hpp"

namespace noarr {

namespace literals {

namespace helpers {

template<std::size_t Accum, char... Chars>
struct idx_translate;

template<std::size_t Accum, char Char, char... Chars>
struct idx_translate<Accum, Char, Chars...> : idx_translate<Accum * 10 + (std::size_t)(Char - '0'), Chars...> {};

template<std::size_t Accum, char Char>
struct idx_translate<Accum, Char> {
	using type = std::integral_constant<std::size_t, Accum * 10 + (std::size_t)(Char - '0')>;
};

} // namespace helpers

/**
 * @brief Converts an integer literal into a corresponding std::integral_constant<std::size_t, ...>
 * 
 * @tparam Chars the digits of the integer literal
 * @return constexpr auto the corresponding std::integral_constant
 */
template<char... Chars>
constexpr auto operator""_idx() noexcept {
	return typename helpers::idx_translate<0, Chars...>::type();
}

}

namespace helpers {

template<class F, class G>
struct compose_impl : contain<F, G> {
	using base = contain<F, G>;

	constexpr compose_impl(F f, G g) noexcept : base(f, g) {}

	template<class T>
	constexpr decltype(auto) operator()(T t) const noexcept {
		return t | base::template get<0>() | base::template get<1>();
	}
};

}

/**
 * @brief composes functions `F` and `G` together
 * 
 * @param f: the inner function (the one applied first)
 * @param g: the outer function
 */
template<class F, class G>
constexpr auto compose(F f, G g) noexcept {
	return helpers::compose_impl<F, G>(f, g);
}


/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim>
constexpr auto set_length(std::size_t length) noexcept {
	return setter(empty_state.with<length_in<Dim>>(length));
}

/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim, std::size_t Length>
constexpr auto set_length(std::integral_constant<std::size_t, Length> length) noexcept {
	return setter(empty_state.with<length_in<Dim>>(length));
}

/**
 * @brief returns the number of indices in the structure specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the desired structure
 */
template<char Dim>
struct get_length {
	explicit constexpr get_length() noexcept {}

	template<class T>
	constexpr std::size_t operator()(T t) const noexcept {
		return spi_length<Dim, T>::get(t, empty_state);
	}
};

/**
 * @brief swaps two structures given by their dimension names in the substructure tree of a structure
 * 
 * @tparam Dim1: the dimension name of the first structure
 * @tparam Dim2: the dimension name of the second structure
 */
template<char Dim1, char Dim2>
struct reassemble; // TODO

template<std::size_t i, class T>
constexpr auto safe_get(T t) noexcept; // TODO

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, class... Ts>
constexpr auto fix(Ts... ts) noexcept {
	return setter(empty_state.with<index_in<Dims>...>(ts...));
}

/**
 * @brief shifts an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for shifting the indices
 */
template<char... Dims, class... Ts>
constexpr auto shift(Ts... ts) noexcept {
	return (unit_struct ^ ... ^ view<Dims>(ts));
}

/**
 * @brief returns the offset of a substructure given by a dimension name in a structure
 * 
 * @tparam Dim: the dimension name
 */
template<char Dim>
constexpr auto get_offset(std::size_t idx) noexcept; // TODO

template<char Dim, std::size_t Idx>
constexpr auto get_offset(std::integral_constant<std::size_t, Idx>) noexcept; // TODO

namespace helpers {

template<class State>
struct offset_impl : contain<State> {
	explicit constexpr offset_impl() noexcept = delete;
	explicit constexpr offset_impl(const State& state) : contain<State>(state) {}

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return spi_offset<T>::get(t, contain<State>::template get<0>());
	}
};

}

/**
 * @brief returns the offset of the value described by the structure
 */
constexpr auto offset() noexcept {
	return helpers::offset_impl(empty_state);
}

/**
 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item
 * 
 * @tparam Dims: the dimension names of fixed indices
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, class... Ts>
constexpr auto offset(Ts... ts) noexcept {
	return helpers::offset_impl(empty_state.with<index_in<Dims>...>(ts...));
}

/**
 * @brief returns the size (in bytes) of the structure
 */
struct get_size {
	constexpr get_size() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return spi_size<T>::get(t, empty_state);
	}
};

namespace helpers {

template<class Ptr, class State>
struct get_at_impl : private contain<Ptr, State> {
	explicit constexpr get_at_impl() noexcept = delete;
	explicit constexpr get_at_impl(Ptr ptr, const State &state) noexcept : contain<Ptr, State>(ptr, state) {}

	template<class T>
	using scalar_type = spi_type_t<T, State>;

	// the return type checks whether the structure `t` is a cube and it also chooses `scalar_t<T> &` or `const scalar_t<T> &` according to constness of `Ptr` pointee
	template<class T>
	constexpr auto operator()(T t) const noexcept -> std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_type<T> &, scalar_type<T> &> {
		// accesses reference to a value with the given offset and casted to its corresponding type
		return *reinterpret_cast<std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_type<T> *, scalar_type<T> *>>(contain<Ptr, State>::template get<0>() + (t | helpers::offset_impl(contain<Ptr, State>::template get<1>())));
	}
};

static inline constexpr char *as_cptr(void *p) noexcept { return (char*)(p); }
static inline constexpr const char *as_cptr(const void *p) noexcept { return (const char*)(p); }

}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 * 
 * @param ptr: the pointer to blob structure
 */
template<class V>
constexpr auto get_at(V *ptr) noexcept {
	return helpers::get_at_impl(helpers::as_cptr(ptr), empty_state);
}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, class V, class... Ts>
constexpr auto get_at(V *ptr, Ts... ts) noexcept {
	return helpers::get_at_impl(helpers::as_cptr(ptr), empty_state.with<index_in<Dims>...>(ts...));
}

/**
 * @brief returns the topmost dims of a structure (if the topmost structure in the substructure tree has no dims and it has only one substructure it returns the topmost dims of this substructure, recursively)
 */
struct top_dims {
	// recursion case for when the topmost structure offers no dims but it has 1 substructure
	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(std::enable_if_t<std::is_same<get_dims<T>, char_pack<>>::value, typename sub_structures<T>::value_type>(std::get<0>(sub_structures<T>(t).value)) | *this) {
		return std::get<0>(sub_structures<T>(t).value) | *this;
	}

	// bottom case
	template<class T>
	constexpr auto operator()(T) const noexcept -> std::enable_if_t<!std::is_same<get_dims<T>, char_pack<>>::value, get_dims<T>> {
		return get_dims<T>();
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_FUNCS_HPP
