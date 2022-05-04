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
template<char Dim, class... TS>
struct tuple;

namespace helpers {
template<class TUPLE, std::size_t I>
struct tuple_part;

template<class T, class... KS>
struct tuple_get_t;

template<char Dim, class T, class... TS, std::size_t I, std::size_t K>
struct tuple_get_t<tuple_part<tuple<Dim, T, TS...>, I>, std::integral_constant<std::size_t, K>> {
	using type = typename tuple_get_t<tuple_part<tuple<Dim, TS...>, I + 1>, std::integral_constant<std::size_t, K - 1>>::type;
};

template<char Dim, class T, class... TS, std::size_t I>
struct tuple_get_t<tuple_part<tuple<Dim, T, TS...>, I>, std::integral_constant<std::size_t, 0>> {
	using type = T;
};

template<char Dim, class T, class... TS, std::size_t I>
struct tuple_part<tuple<Dim, T, TS...>, I> : contain<T, tuple_part<tuple<Dim, TS...>, I + 1>> {
	template<class, std::size_t>
	friend struct tuple_part;

	constexpr tuple_part() noexcept = default;
	explicit constexpr tuple_part(T t, TS... ts) noexcept : base(t, tuple_part<tuple<Dim, TS...>, I + 1>(ts...)) {}

protected:
	using base = contain<T, tuple_part<tuple<Dim, TS...>, I + 1>>;

	constexpr auto sub_structures() const noexcept {
		return std::tuple_cat(std::tuple<T>(base::template get<0>()), base::template get<1>().sub_structures());
	}
};

template<char Dim, class T, std::size_t I>
struct tuple_part<tuple<Dim, T>, I> : contain<T> {
	template<class, std::size_t>
	friend struct tuple_part;

	constexpr tuple_part() noexcept = default;
	explicit constexpr tuple_part(T t) noexcept : contain<T>(t) {}

protected:
	constexpr auto sub_structures() const noexcept {
		return std::tuple<T>(contain<T>::template get<0>());
	}
};

template<class T, std::size_t I = std::tuple_size<T>::value>
struct tuple_size_getter;

template<class T, std::size_t I>
struct tuple_size_getter {
	template<std::size_t... IS>
	static constexpr std::size_t size(T t) noexcept {
		return tuple_size_getter<T, I - 1>::template size<I - 1, IS...>(t);
	}
};

template<class T>
struct tuple_size_getter<T, 0> {
	template<class... Args>
	static constexpr std::size_t sum(std::size_t arg, Args... args) noexcept {
		return arg + sum(args...);
	}

	static constexpr std::size_t sum(std::size_t arg) noexcept {
		return arg;
	}

	template<std::size_t... IS>
	static constexpr std::size_t size(T t) noexcept {
		return sum(std::get<IS>(t).size()...);
	}

	static constexpr std::size_t size(T) noexcept {
		return 0;
	}
};

}

template<char Dim, class T, class... TS>
struct tuple<Dim, T, TS...> : helpers::tuple_part<tuple<Dim, T, TS...>, 0> {
	constexpr std::tuple<T, TS...> sub_structures() const noexcept { return helpers::tuple_part<tuple<Dim, T, TS...>, 0>::sub_structures(); }
	using description = struct_description<
		char_pack<'t', 'u', 'p', 'l', 'e'>,
		dims_impl<Dim>,
		dims_impl<>,
		structure_param<T>,
		structure_param<TS>...>;

	template<class... KS>
	using get_t = typename helpers::tuple_get_t<helpers::tuple_part<tuple<Dim, T, TS...>, 0>, KS...>::type;

	constexpr tuple() noexcept = default;
	constexpr tuple(T ss, TS... sss) noexcept : helpers::tuple_part<tuple<Dim, T, TS...>, 0>(ss, sss...) {}

	template<class T2, class... T2s>
	static constexpr auto construct(T2 ss, T2s... sss) noexcept {
		return tuple<Dim, T2, T2s...>(ss, sss...);
	}

	constexpr std::size_t size() const noexcept {
		return helpers::tuple_size_getter<remove_cvref<decltype(sub_structures())>>::size(sub_structures());
	}
	template<std::size_t i>
	constexpr std::size_t offset() const noexcept {
		return helpers::tuple_size_getter<remove_cvref<decltype(sub_structures())>, i>::size(sub_structures());
	}
	static constexpr std::size_t length() noexcept { return sizeof...(TS) + 1; }
};

namespace helpers {

template<class T, class... KS>
struct array_get_t;

template<class T>
struct array_get_t<T> {
	using type = T;
};

template<class T>
struct array_get_t<T, void> {
	using type = T;
};

template<class T, std::size_t K>
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
template<char Dim, std::size_t L, class T = void>
struct array : contain<T> {
	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'a', 'r', 'r', 'a', 'y'>,
		dims_impl<Dim>,
		dims_impl<>,
		value_param<std::size_t, L>,
		structure_param<T>>;

	template<class... KS>
	using get_t = typename helpers::array_get_t<T, KS...>::type;

	constexpr array() noexcept = default;
	explicit constexpr array(T sub_structure) noexcept : contain<T>(sub_structure) {}

	template<class T2>
	static constexpr auto construct(T2 sub_structure) noexcept {
		return array<Dim, L, T2>(sub_structure);
	}

	constexpr std::size_t size() const noexcept { return contain<T>::template get<0>().size() * L; }
	constexpr std::size_t offset(std::size_t i) const noexcept { return contain<T>::template get<0>().size() * i; }
	template<std::size_t I>
	constexpr std::size_t offset() const noexcept { return std::get<0>(sub_structures()).size() * I; }
	static constexpr std::size_t length() noexcept { return L; }
};

template<char Dim, std::size_t L>
struct array<Dim, L> {
	constexpr std::tuple<> sub_structures() const noexcept { return {}; }
	using description = struct_description<
		char_pack<'a', 'r', 'r', 'a', 'y'>,
		dims_impl<Dim>,
		dims_impl<>,
		value_param<std::size_t, L>>;

	constexpr array() noexcept = default;

	template<class T2>
	static constexpr auto construct(T2 sub_structure) noexcept {
		return array<Dim, L, T2>(sub_structure);
	}

	static constexpr auto construct() noexcept {
		return array();
	}


	static constexpr std::size_t length() noexcept { return L; }
};

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 * 
 * @tparam Dim: the dimmension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<char Dim, class T = void>
struct vector : contain<T> {
	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(contain<T>::template get<0>()); }
	using description = struct_description<
		char_pack<'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
		structure_param<T>>;

	constexpr vector() noexcept = default;
	explicit constexpr vector(T sub_structure) noexcept : contain<T>(sub_structure) {}

	template<class T2>
	static constexpr auto construct(T2 sub_structure) noexcept {
		return vector<Dim, T2>(sub_structure);
	}
};


template<char Dim>
struct vector<Dim> {
	constexpr std::tuple<> sub_structures() const noexcept { return {}; }
	using description = struct_description<
		char_pack<'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>>;

	constexpr vector() noexcept = default;

	template<class T2>
	static constexpr auto construct(T2 sub_structure) noexcept {
		return vector<Dim, T2>(sub_structure);
	}

	static constexpr auto construct() noexcept {
		return vector();
	}
};

namespace helpers {

template<class T, class... KS>
struct sized_vector_get_t;

template<class T>
struct sized_vector_get_t<T> {
	using type = T;
};

template<class T>
struct sized_vector_get_t<T, void> {
	using type = T;
};

template<class T, std::size_t K>
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
template<char Dim, class T = void>
struct sized_vector : contain<vector<Dim, T>, std::size_t> {
	using base = contain<vector<Dim, T>, std::size_t>;
	constexpr std::tuple<T> sub_structures() const noexcept { return base::template get<0>().sub_structures(); }
	using description = struct_description<
		char_pack<'s', 'i', 'z', 'e', 'd', '_', 'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
		structure_param<T>>;

	template<class... KS>
	using get_t = typename helpers::sized_vector_get_t<T, KS...>::type;

	constexpr sized_vector() noexcept = default;
	constexpr sized_vector(T sub_structure, std::size_t length) noexcept : base(vector<Dim, T>(sub_structure), length) {}

	template<class T2>
	constexpr auto construct(T2 sub_structure) const noexcept {
		return sized_vector<Dim, T2>(sub_structure, base::template get<1>());
	}

	constexpr std::size_t size() const noexcept { return std::get<0>(sub_structures()).size() * base::template get<1>(); }
	constexpr std::size_t offset(std::size_t i) const noexcept { return std::get<0>(sub_structures()).size() * i; }
	template<std::size_t I>
	constexpr std::size_t offset() const noexcept { return std::get<0>(sub_structures()).size() * I; }
	constexpr std::size_t length() const noexcept { return base::template get<1>(); }
};

template<char Dim>
struct sized_vector<Dim> : contain<std::size_t> {
	using base = contain<std::size_t>;
	constexpr std::tuple<> sub_structures() const noexcept { return {}; }
	using description = struct_description<
		char_pack<'s', 'i', 'z', 'e', 'd', '_', 'v', 'e', 'c', 't', 'o', 'r'>,
		dims_impl<Dim>,
		dims_impl<>,
		structure_param<void>>;

	constexpr sized_vector() noexcept = default;
	constexpr sized_vector(std::size_t length) noexcept : base(length) {}

	template<class T2>
	constexpr auto construct(T2 sub_structure) const noexcept {
		return sized_vector<Dim, T2>(sub_structure, base::template get<0>());
	}

	constexpr auto construct() const noexcept {
		return sized_vector(base::template get<0>());
	}

	constexpr std::size_t length() const noexcept { return base::template get<0>(); }
};

template<class Struct, class ProtoStruct>
constexpr auto operator^(Struct &&s, ProtoStruct &&p)  -> decltype(p.construct(std::forward<Struct>(s))) {
	return p.construct(std::forward<Struct>(s));
}

namespace helpers {

template<class T, std::size_t Idx, class... KS>
struct sfixed_dim_get_t;

template<class T, std::size_t Idx>
struct sfixed_dim_get_t<T, Idx> {
	using type = typename T::template get_t<std::integral_constant<std::size_t, Idx>>;
};

template<class T, std::size_t Idx>
struct sfixed_dim_get_t<T, Idx, void> {
	using type = typename T::template get_t<std::integral_constant<std::size_t, Idx>>;
};

}

template<class T, class = void>
struct is_static_construct : std::false_type {};

template<class T>
struct is_static_construct<T, decltype(&T::construct, void())> : std::true_type {};

/**
 * @brief constant fixed dimension, carries a single sub_structure with a statically fixed index
 * 
 * @tparam T: substructure type
 */
template<char Dim, class T, std::size_t Idx>
struct sfixed_dim : contain<T> {
	/* e.g. sub_structures of a sfixed tuple are the same as the substructures of the tuple
	 *(otherwise we could not have the right offset after an item with Idx2 < Idx in the tuple changes)
	 */
	constexpr auto sub_structures() const noexcept { return contain<T>::template get<0>().sub_structures(); }
	using description = struct_description<
		char_pack<'s', 'f', 'i', 'x', 'e', 'd', '_', 'd', 'i', 'm'>,
		dims_impl<>,
		dims_impl<Dim>,
		structure_param<T>,
		value_param<std::size_t, Idx>>;

	template<class... KS>
	using get_t = typename helpers::sfixed_dim_get_t<T, Idx, KS...>::type;

	constexpr sfixed_dim() noexcept = default;
	explicit constexpr sfixed_dim(T sub_structure) noexcept : contain<T>(sub_structure) {}
	
	template<class T2, class... T2s>
	constexpr auto construct(T2 ss, T2s... sss) const noexcept {
		return sfixed_dim<Dim, decltype(std::declval<T>().construct(ss, sss...)), Idx>(
			contain<T>::template get<0>().construct(ss, sss...));
	}

	constexpr std::size_t size() const noexcept { return contain<T>::template get<0>().size(); }
	constexpr std::size_t offset() const noexcept { return contain<T>::template get<0>().template offset<Idx>(); }
	constexpr std::size_t length() const noexcept { return 0; }
};

namespace helpers {

template<class T, class... KS>
struct fixed_dim_get_t;

template<class T>
struct fixed_dim_get_t<T> {
	using type = typename T::template get_t<>;
};

template<class T>
struct fixed_dim_get_t<T, void> {
	using type = typename T::template get_t<void>;
};

}

/**
 * @brief fixed dimension, carries a single sub_structure with a fixed index
 * 
 * @tparam T: substructure type
 */
template<char Dim, class T>
struct fixed_dim : contain<T, std::size_t> {
	using base = contain<T, std::size_t>;
	constexpr auto sub_structures() const noexcept { return noarr::sub_structures<decltype(this->base::template get<0>())>(base::template get<0>()).value; }
	using description = struct_description<
		char_pack<'f', 'i', 'x', 'e', 'd', '_', 'd', 'i', 'm'>,
		dims_impl<>,
		dims_impl<Dim>,
		structure_param<T>>;

	template<class... KS>
	using get_t = typename helpers::fixed_dim_get_t<T, KS...>::type;

	constexpr fixed_dim() noexcept = default;
	constexpr fixed_dim(T sub_structure, std::size_t idx) noexcept : base(sub_structure, idx) {}

	template<class... T2>
	constexpr auto construct(T2...sub_structures) const noexcept {
		return fixed_dim<Dim, decltype(std::declval<T>().construct(sub_structures...))>(
			base::template get<0>().construct(sub_structures...),
			base::template get<1>());
	}

	constexpr std::size_t size() const noexcept { return base::template get<0>().size(); }
	constexpr std::size_t offset() const noexcept { return base::template get<0>().offset(base::template get<1>()); }
	constexpr std::size_t length() const noexcept { return 0; }
};

namespace helpers {

template<class T, class... KS>
struct shifted_dim_get_t;

template<class T>
struct shifted_dim_get_t<T> {
	using type = typename T::template get_t<>;
};

template<class T>
struct shifted_dim_get_t<T, void> {
	using type = typename T::template get_t<void>;
};

}

/**
 * @brief shifted dimension, carries a single sub_structure with a shifted index
 * 
 * @tparam T: substructure type
 */
template<char Dim, class T>
struct shifted_dim : contain<T, std::size_t> {
	using base = contain<T, std::size_t>;
	constexpr auto sub_structures() const noexcept { return noarr::sub_structures<decltype(this->base::template get<0>())>(base::template get<0>()).value; }
	using description = struct_description<
		char_pack<'s', 'h', 'i', 'f', 't', 'e', 'd', '_', 'd', 'i', 'm'>,
		dims_impl<Dim>,
		dims_impl<Dim>,
		structure_param<T>>;

	template<class... KS>
	using get_t = typename helpers::shifted_dim_get_t<T, KS...>::type;

	constexpr shifted_dim() noexcept = default;
	constexpr shifted_dim(T sub_structure, std::size_t idx) noexcept : base(sub_structure, idx) {}

	template<class... T2>
	constexpr auto construct(T2...sub_structures) const noexcept {
		return shifted_dim<Dim, decltype(std::declval<T>().construct(sub_structures...))>(
			base::template get<0>().construct(sub_structures...),
			base::template get<1>());
	}

	constexpr std::size_t size() const noexcept { return base::template get<0>().size(); }
	constexpr std::size_t offset(std::size_t i) const noexcept { return base::template get<0>().offset(i + base::template get<1>()); }
	template<std::size_t I>
	constexpr std::size_t offset() const noexcept { return base::template get<0>().offset(I + base::template get<1>()); }
	constexpr std::size_t length() const noexcept { return base::template get<0>().length() - base::template get<1>(); }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_STRUCTS_HPP
