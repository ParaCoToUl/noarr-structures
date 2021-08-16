#ifndef NOARR_STRUCTURES_STRUCTS_HPP
#define NOARR_STRUCTURES_STRUCTS_HPP

#include "struct_decls.hpp"
#include "contain.hpp"
#include "scalar.hpp"

namespace noarr {

/**
 * @brief tuple
 * 
 * @tparam Dim dimmension added by the structure
 * @tparam T,TS... substructure types
 */
template<char Dim, typename... TS>
struct tuple;

namespace helpers {

template<typename TUPLE, std::size_t I>
struct tuple_part;

template<typename T, typename... KS>
struct tuple_get_t;

template<char Dim, typename T, typename... TS, std::size_t I, std::size_t K>
struct tuple_get_t<tuple_part<tuple<Dim, T, TS...>, I>, std::integral_constant<std::size_t, K>> {
	using type = typename tuple_get_t<tuple_part<tuple<Dim, TS...>, I + 1>, std::integral_constant<std::size_t, K - 1>>::type;
};

template<char Dim, typename T, typename... TS, std::size_t I>
struct tuple_get_t<tuple_part<tuple<Dim, T, TS...>, I>, std::integral_constant<std::size_t, 0>> {
	using type = T;
};

template<char Dim, typename T, typename... TS, std::size_t I>
struct tuple_part<tuple<Dim, T, TS...>, I> : private contain<T, tuple_part<tuple<Dim, TS...>, I + 1>> {
	using base = contain<T, tuple_part<tuple<Dim, TS...>, I + 1>>;

	constexpr tuple_part() = default;
	explicit constexpr tuple_part(T t, TS... ts) : base(t, tuple_part<tuple<Dim, TS...>, I + 1>(ts...)) {}

	constexpr auto sub_structures() const {
		return std::tuple_cat(std::tuple<T>(base::template get<0>()), base::template get<1>().sub_structures());
	}
};

template<char Dim, typename T, std::size_t I>
struct tuple_part<tuple<Dim, T>, I> : private contain<T> {
	constexpr tuple_part() = default;
	explicit constexpr tuple_part(T t) : contain<T>(t) {}

	constexpr auto sub_structures() const {
		return std::tuple<T>(contain<T>::template get<0>());
	}
};

template<typename T, std::size_t I = std::tuple_size<T>::value>
struct tuple_size_getter;

template<typename T, std::size_t I>
struct tuple_size_getter {
	template<std::size_t... IS>
	static constexpr std::size_t size(T t) {
		return tuple_size_getter<T, I - 1>::template size<I - 1, IS...>(t);
	}
};

template<typename T>
struct tuple_size_getter<T, 0> {
	template<typename... Args>
	static constexpr std::size_t sum(std::size_t arg, Args... args) {
		return arg + sum(args...);
	}

	static constexpr std::size_t sum(std::size_t arg) {
		return arg;
	}

	template<std::size_t... IS>
	static constexpr std::size_t size(T t) {
		return sum(std::get<IS>(t).size()...);
	}

	static constexpr std::size_t size(T) {
		return 0;
	}
};

}

template<char Dim, typename T, typename... TS>
struct tuple<Dim, T, TS...> : private helpers::tuple_part<tuple<Dim, T, TS...>, 0> {
	constexpr std::tuple<T, TS...> sub_structures() const { return helpers::tuple_part<tuple<Dim, T, TS...>, 0>::sub_structures(); }
	using description = struct_description<
		char_pack<'t', 'u', 'p', 'l', 'e'>,
		dims_impl<Dim>,
		dims_impl<>,
		type_param<T>,
		type_param<TS>...>;

	template<typename... KS>
	using get_t = typename helpers::tuple_get_t<helpers::tuple_part<tuple<Dim, T, TS...>, 0>, KS...>::type;

	constexpr tuple() = default;
	constexpr tuple(T ss, TS... sss) : helpers::tuple_part<tuple<Dim, T, TS...>, 0>(ss, sss...) {}

	template<typename T2, typename... T2s>
	static constexpr auto construct(T2 ss, T2s... sss) {
		return tuple<Dim, T2, T2s...>(ss, sss...);
	}

	constexpr std::size_t size() const {
		return helpers::tuple_size_getter<remove_cvref<decltype(sub_structures())>>::size(sub_structures());
	}
	template<std::size_t i>
	constexpr std::size_t offset() const {
		return helpers::tuple_size_getter<remove_cvref<decltype(sub_structures())>, i>::size(sub_structures());
	}
	static constexpr std::size_t length() { return sizeof...(TS) + 1; }
};

namespace helpers {

template<typename T, typename... KS>
struct array_get_t;

template<typename T>
struct array_get_t<T> {
	using type = T;
};

template<typename T>
struct array_get_t<T, void> {
	using type = T;
};

template<typename T, std::size_t K>
struct array_get_t<T, std::integral_constant<std::size_t, K>> {
	using type = T;
};

}

/**
 * @brief a structure representing an array with a dynamicly specifiable index (all indices point to the same substructure, with a different offset)
 * 
 * @tparam Dim: the dimmension name added by the array
 * @tparam T: the type of the substructure the array contains
 */
template<char Dim, std::size_t L, typename T>
struct array : private contain<T> {
	constexpr std::tuple<T> sub_structures() const { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'a', 'r', 'r', 'a', 'y'>,
		dims_impl<Dim>,
		dims_impl<>,
		value_param<std::size_t, L>,
		type_param<T>>;

	template<typename... KS>
	using get_t = typename helpers::array_get_t<T, KS...>::type;

	constexpr array() = default;
	explicit constexpr array(T sub_structure) : contain<T>(sub_structure) {}

	template<typename T2>
	static constexpr auto construct(T2 sub_structure) {
		return array<Dim, L, T2>(sub_structure);
	}

	constexpr std::size_t size() const { return contain<T>::template get<0>().size() * L; }
	constexpr std::size_t offset(std::size_t i) const { return contain<T>::template get<0>().size() * i; }
	template<std::size_t I>
	constexpr std::size_t offset() const { return std::get<0>(sub_structures()).size() * I; }
	static constexpr std::size_t length() { return L; }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 * 
 * @tparam Dim: the dimmension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<char Dim, typename T>
struct vector : private contain<T> {
	constexpr std::tuple<T> sub_structures() const { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
		type_param<T>>;

	constexpr vector() = default;
	explicit constexpr vector(T sub_structure) : contain<T>(sub_structure) {}

	template<typename T2>
	static constexpr auto construct(T2 sub_structure) {
		return vector<Dim, T2>(sub_structure);
	}
};

namespace helpers {

template<typename T, typename... KS>
struct sized_vector_get_t;

template<typename T>
struct sized_vector_get_t<T> {
	using type = T;
};

template<typename T>
struct sized_vector_get_t<T, void> {
	using type = T;
};

template<typename T, std::size_t K>
struct sized_vector_get_t<T, std::integral_constant<std::size_t, K>> {
	using type = T;
};

}

/**
 * @brief sized vector (size reassignable by the resize function), see `vector`
 * 
 * @tparam Dim: the dimmension name added by the sized vector
 * @tparam T: the type of the substructure the sized vector consists of
 */
template<char Dim, typename T>
struct sized_vector : private contain<vector<Dim, T>, std::size_t> {
	using base = contain<vector<Dim, T>, std::size_t>;
	constexpr std::tuple<T> sub_structures() const { return base::template get<0>().sub_structures(); }
	using description = struct_description<
		char_pack<'s', 'i', 'z', 'e', 'd', '_', 'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
		type_param<T>>;

	template<typename... KS>
	using get_t = typename helpers::sized_vector_get_t<T, KS...>::type;

	constexpr sized_vector() = default;
	constexpr sized_vector(T sub_structure, std::size_t length) : base(vector<Dim, T>(sub_structure), length) {}

	template<typename T2>
	constexpr auto construct(T2 sub_structure) const {
		return sized_vector<Dim, T2>(sub_structure, base::template get<1>());
	}

	constexpr std::size_t size() const { return std::get<0>(sub_structures()).size() * base::template get<1>(); }
	constexpr std::size_t offset(std::size_t i) const { return std::get<0>(sub_structures()).size() * i; }
	template<std::size_t I>
	constexpr std::size_t offset() const { return std::get<0>(sub_structures()).size() * I; }
	constexpr std::size_t length() const { return base::template get<1>(); }
};

namespace helpers {

template<typename T, std::size_t Idx, typename... KS>
struct sfixed_dim_get_t;

template<typename T, std::size_t Idx>
struct sfixed_dim_get_t<T, Idx> {
	using type = typename T::template get_t<std::integral_constant<std::size_t, Idx>>;
};

template<typename T, std::size_t Idx>
struct sfixed_dim_get_t<T, Idx, void> {
	using type = typename T::template get_t<std::integral_constant<std::size_t, Idx>>;
};

}

template<typename T, typename = void>
struct is_static_construct {
	static constexpr bool value = false;
};

template<typename T>
struct is_static_construct<T, decltype(&T::construct, void())> {
	static constexpr bool value = true;
};

/**
 * @brief constant fixed dimension, carries a single sub_structure with a statically fixed index
 * 
 * @tparam T: substructure type
 */
template<char Dim, typename T, std::size_t Idx>
struct sfixed_dim : private contain<T> {
	/* e.g. sub_structures of a sfixed tuple are the same as the substructures of the tuple
	 *(otherwise we could not have the right offset after an item with Idx2 < Idx in the tuple changes)
	 */
	constexpr auto sub_structures() const { return contain<T>::template get<0>().sub_structures(); }
	using description = struct_description<
		char_pack<'s', 'f', 'i', 'x', 'e', 'd', '_', 'd', 'i', 'm'>,
		dims_impl<>,
		dims_impl<Dim>,
		type_param<T>,
		value_param<std::size_t, Idx>>;

	template<typename... KS>
	using get_t = typename helpers::sfixed_dim_get_t<T, Idx, KS...>::type;

	constexpr sfixed_dim() = default;
	explicit constexpr sfixed_dim(T sub_structure) : contain<T>(sub_structure) {}
	
	template<typename T2, typename... T2s>
	constexpr auto construct(T2 ss, T2s... sss) const {
		return sfixed_dim<Dim, decltype(std::declval<T>().construct(ss, sss...)), Idx>(
			contain<T>::template get<0>().construct(ss, sss...));
	}

	constexpr std::size_t size() const { return contain<T>::template get<0>().size(); }
	constexpr std::size_t offset() const { return contain<T>::template get<0>().template offset<Idx>(); }
	constexpr std::size_t length() const { return 0; }
};

namespace helpers {

template<typename T, typename... KS>
struct fixed_dim_get_t;

template<typename T>
struct fixed_dim_get_t<T> {
	using type = T;
};

template<typename T>
struct fixed_dim_get_t<T, void> {
	using type = T;
};

}

/**
 * @brief fixed dimension, carries a single sub_structure with a fixed index
 * 
 * @tparam T: substructure type
 */
template<char Dim, typename T>
struct fixed_dim : private contain<T, std::size_t> {
	using base = contain<T, std::size_t>;
	constexpr auto sub_structures() const { return noarr::sub_structures<decltype(this->base::template get<0>())>(base::template get<0>()).value; }
	using description = struct_description<
		char_pack<'f', 'i', 'x', 'e', 'd', '_', 'd', 'i', 'm'>,
		dims_impl<>,
		dims_impl<Dim>,
		type_param<T>>;

	template<typename... KS>
	using get_t = typename helpers::fixed_dim_get_t<T, KS...>::type;

	constexpr fixed_dim() = default;
	constexpr fixed_dim(T sub_structure, std::size_t idx) : base(sub_structure, idx) {}

	template<typename... T2>
	constexpr auto construct(T2...sub_structures) const {
		return fixed_dim<Dim, decltype(std::declval<T>().construct(sub_structures...))>(
			base::template get<0>().construct(sub_structures...),
			base::template get<1>());
	}

	constexpr std::size_t size() const { return base::template get<0>().size(); }
	constexpr std::size_t offset() const { return base::template get<0>().offset(base::template get<1>()); }
	constexpr std::size_t length() const { return 0; }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCTS_HPP