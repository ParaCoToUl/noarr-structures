#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "struct_traits.hpp"
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
	// the composed functions are applied on the structure as a whole (it does not inherit the func_family)
	// the `func_family`s of the composed functions are still relevant
	// (the operator() calls the functions as if they were applied to the given structure directly)
	using func_family = top_tag;

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

namespace helpers {

template<char Dim, class T>
struct dynamic_set_length_can_apply : std::false_type {};

template<char Dim, class T>
struct dynamic_set_length_can_apply<Dim, vector<Dim, T>> : std::true_type {};

template<char Dim, class T>
struct dynamic_set_length_can_apply<Dim, sized_vector<Dim, T>> : std::true_type {};

}

namespace helpers {

template<char Dim>
struct dynamic_set_length {
	using func_family = transform_tag;

	template<class T>
	using can_apply = helpers::dynamic_set_length_can_apply<Dim, T>;

	explicit constexpr dynamic_set_length(std::size_t length) noexcept : length(length) {}

	template<class T>
	constexpr auto operator()(vector<Dim, T> v) const noexcept {
		return sized_vector<Dim, T>(std::get<0>(v.sub_structures()), length);
	}

	template<class T>
	constexpr auto operator()(sized_vector<Dim, T> v) const noexcept {
		return sized_vector<Dim, T>(std::get<0>(v.sub_structures()), length);
	}

private:
	std::size_t length;
};

template<char Dim, std::size_t L>
struct static_set_length_impl {
	using func_family = transform_tag;

	constexpr static_set_length_impl() noexcept = default;

	template<class T>
	constexpr auto operator()(vector<Dim, T> v) const noexcept {
		return array<Dim, L, T>(std::get<0>(v.sub_structures()));
	}

	template<class T>
	constexpr auto operator()(sized_vector<Dim, T> v) const noexcept {
		return array<Dim, L, T>(std::get<0>(v.sub_structures()));
	}

	template<class T, std::size_t L2>
	constexpr auto operator()(array<Dim, L2, T> v) const noexcept {
		return array<Dim, L, T>(std::get<0>(v.sub_structures()));
	}
};

}

/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim>
constexpr auto set_length(std::size_t length) noexcept {
	return helpers::dynamic_set_length<Dim>(length);
}

/**
 * @brief sets the length of a `vector`, `sized_vector` or an `array` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char Dim, std::size_t Length>
constexpr auto set_length(std::integral_constant<std::size_t, Length>) noexcept {
	return helpers::static_set_length_impl<Dim, Length>();
}

/**
 * @brief returns the number of indices in the structure specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the desired structure
 */
template<char Dim>
struct get_length {
	using func_family = get_tag;

	template<class T>
	using can_apply = typename get_dims<T>::template contains<Dim>;

	explicit constexpr get_length() noexcept {}

	template<class T>
	constexpr std::size_t operator()(T t) const noexcept {
		return t.length();
	}
};

namespace helpers {

template<char Dim>
struct reassemble_get {
	using func_family = get_tag;

	template<class T>
	using can_apply = typename get_dims<T>::template contains<Dim>;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t;
	}
};

template<char Dim, class T, class T2>
struct reassemble_set : private contain<T> {
	using func_family = transform_tag;

	constexpr reassemble_set() noexcept = default;
	explicit constexpr reassemble_set(T t) noexcept : contain<T>(t) {}

	constexpr auto operator()(T2 t) const noexcept {
		return construct(contain<T>::template get<0>(), t.sub_structures());
	}
};

}

/**
 * @brief swaps two structures given by their dimension names in the substructure tree of a structure
 * 
 * @tparam Dim1: the dimension name of the first structure
 * @tparam Dim2: the dimension name of the second structure
 */
template<char Dim1, char Dim2>
struct reassemble {
private:
	template<char Dim, class T, class T2>
	constexpr auto add_setter(T t, T2 t2) const noexcept {
		return construct(t2, (t | helpers::reassemble_set<Dim, T, remove_cvref<decltype(t2)>>(t)).sub_structures());
	}

	template<char Dim, class T>
	constexpr auto add_getter(T t) const noexcept -> decltype(add_setter<Dim>(t, t | helpers::reassemble_get<Dim>())) {
		return add_setter<Dim>(t, t | helpers::reassemble_get<Dim>());
	}

public:
	using func_family = transform_tag;

	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(add_getter<std::enable_if_t<get_dims<T>::template contains<Dim1>::value, char>(Dim2)>(t)) {
		return add_getter<Dim2>(t);
	}

	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(add_getter<std::enable_if_t<get_dims<T>::template contains<Dim2>::value && Dim1 != Dim2, char>(Dim1)>(t)) {
		return add_getter<Dim1>(t);
	}
};

namespace helpers {

template<class T, std::size_t i, class = void>
struct safe_get_impl {
	static constexpr void get(T t) noexcept = delete;
};

template<class T, std::size_t i>
struct safe_get_impl<T, i, std::enable_if_t<(std::tuple_size<remove_cvref<decltype(std::declval<T>().sub_structures())>>::value > i)>> {
	static constexpr auto get(T t) noexcept {
		return std::get<i>(t.sub_structures());
	}
};

}

template<std::size_t i, class T>
constexpr auto safe_get(T t) noexcept {
	return helpers::safe_get_impl<T, i>::get(t);
}

namespace helpers {

template<char Dim>
struct fix_dynamic_impl {
	using func_family = transform_tag;

	constexpr fix_dynamic_impl() noexcept = default;
	explicit constexpr fix_dynamic_impl(std::size_t idx) noexcept : idx(idx) {}

private:
	std::size_t idx;

public:
	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), fixed_dim<Dim, T>(t, idx)) {
		return fixed_dim<Dim, T>(t, idx);
	}
};

template<char Dim, std::size_t Idx>
struct fix_static_impl {
	using func_family = transform_tag;
	using idx_t = std::integral_constant<std::size_t, Idx>;

	constexpr fix_static_impl() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), sfixed_dim<Dim, T, Idx>(t)) {
		return sfixed_dim<Dim, T, Idx>(t);
	}
};

template<class... Tuples>
struct fix_impl;

template<char Dim, class T, class... Tuples>
struct fix_impl<std::tuple<std::integral_constant<char, Dim>, T>, Tuples...> : private contain<fix_dynamic_impl<Dim>, fix_impl<Tuples...>> {
	using base = contain<fix_dynamic_impl<Dim>, fix_impl<Tuples...>>;
	using func_family = transform_tag;

	constexpr fix_impl() noexcept = default;
	
	template <class... Ts>
	constexpr fix_impl(T t, Ts... ts) noexcept : base(fix_dynamic_impl<Dim>(t), fix_impl<Tuples...>(ts...)) {}

	template<class S>
	constexpr auto operator()(S s) const noexcept {
		return pipe(s, base::template get<0>(), base::template get<1>());
	}
};

template<char Dim, class T>
struct fix_impl<std::tuple<std::integral_constant<char, Dim>, T>> : private fix_dynamic_impl<Dim> {
	using func_family = transform_tag;
	using fix_dynamic_impl<Dim>::fix_dynamic_impl;
	using fix_dynamic_impl<Dim>::operator();
};


template<char Dim, std::size_t Idx, class... Tuples>
struct fix_impl<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>, Tuples...> : private contain<fix_static_impl<Dim, Idx>, fix_impl<Tuples...>> {
	using base = contain<fix_static_impl<Dim, Idx>, fix_impl<Tuples...>>;
	using func_family = transform_tag;

	constexpr fix_impl() noexcept = default;
	
	template <class... Ts>
	constexpr fix_impl(std::integral_constant<std::size_t, Idx>, Ts... ts) noexcept : base(fix_static_impl<Dim, Idx>(), fix_impl<Tuples...>(ts...)) {}

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return pipe(t, base::template get<0>(), base::template get<1>());
	}
};

template<char Dim, std::size_t Idx>
struct fix_impl<std::tuple<std::integral_constant<char, Dim>, std::integral_constant<std::size_t, Idx>>> : private fix_static_impl<Dim, Idx> {
	using func_family = transform_tag;

	constexpr fix_impl() noexcept = default;
	constexpr fix_impl(std::integral_constant<std::size_t, Idx>) noexcept : fix_static_impl<Dim,Idx>() {}

	using fix_static_impl<Dim, Idx>::operator();
};

template<>
struct fix_impl<> {
	using func_family = transform_tag;

	constexpr fix_impl() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t;
	}
};

template<class... Tuples>
struct shift_impl;

template<char Dim>
struct shift_dynamic_impl {
	using func_family = transform_tag;

	constexpr shift_dynamic_impl() noexcept = default;
	explicit constexpr shift_dynamic_impl(std::size_t idx) noexcept : idx(idx) {}

private:
	std::size_t idx;

public:
	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(std::declval<std::enable_if_t<get_dims<T>::template contains<Dim>::value>>(), shifted_dim<Dim, T>(t, idx)) {
		return shifted_dim<Dim, T>(t, idx);
	}
};

template<char Dim, class T, class... Tuples>
struct shift_impl<std::tuple<std::integral_constant<char, Dim>, T>, Tuples...> : private contain<shift_dynamic_impl<Dim>, shift_impl<Tuples...>> {
	using base = contain<shift_dynamic_impl<Dim>, shift_impl<Tuples...>>;
	using func_family = transform_tag;

	constexpr shift_impl() noexcept = default;
	
	template <class... Ts>
	constexpr shift_impl(T t, Ts... ts) noexcept : base(shift_dynamic_impl<Dim>(t), shift_impl<Tuples...>(ts...)) {}

	template<class S>
	constexpr auto operator()(S s) const noexcept {
		return pipe(s, base::template get<0>(), base::template get<1>());
	}
};

template<char Dim, class T>
struct shift_impl<std::tuple<std::integral_constant<char, Dim>, T>> : private shift_dynamic_impl<Dim> {
	using func_family = transform_tag;
	using shift_dynamic_impl<Dim>::shift_dynamic_impl;
	using shift_dynamic_impl<Dim>::operator();
};

template<>
struct shift_impl<> {
	using func_family = transform_tag;

	constexpr shift_impl() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t;
	}
};

}

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, class... Ts>
constexpr auto fix(Ts... ts) noexcept {
	return helpers::fix_impl<std::tuple<std::integral_constant<char, Dims>, Ts>...>(ts...);
}

/**
 * @brief shifts an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for shifting the indices
 */
template<char... Dims, class... Ts>
constexpr auto shift(Ts... ts) noexcept {
	return helpers::shift_impl<std::tuple<std::integral_constant<char, Dims>, Ts>...>(ts...);
}

namespace helpers {

template<char Dim>
struct get_offset_dynamic_impl {
	using func_family = get_tag;

	explicit constexpr get_offset_dynamic_impl(std::size_t idx) noexcept : idx(idx) {}

	template<class T>
	using can_apply = typename get_dims<T>::template contains<Dim>;

private:
	std::size_t idx;

public:
	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t.offset(idx);
	}
};

template<char Dim, std::size_t Idx>
struct get_offset_static_impl {
	using func_family = get_tag;

	template<class T>
	using can_apply = typename get_dims<T>::template contains<Dim>;

	template<class T>
	constexpr auto operator()(T t) const noexcept {
		return t.template offset<Idx>();
	}
};

}

/**
 * @brief returns the offset of a substructure given by a dimension name in a structure
 * 
 * @tparam Dim: the dimension name
 */
template<char Dim>
constexpr auto get_offset(std::size_t idx) noexcept {
	return helpers::get_offset_dynamic_impl<Dim>(idx);
}

template<char Dim, std::size_t Idx>
constexpr auto get_offset(std::integral_constant<std::size_t, Idx>) noexcept {
	return helpers::get_offset_static_impl<Dim, Idx>();
}

namespace helpers {

struct offset_impl {
	using func_family = top_tag;
	explicit constexpr offset_impl() noexcept = default;

	template<class T>
	constexpr std::size_t operator()(scalar<T>) const noexcept {
		return 0;
	}

	template<class T>
	constexpr auto operator()(T t) const noexcept -> std::enable_if_t<is_point<T>::value, std::size_t> {
		return t.offset() + (std::get<0>(t.sub_structures()) | offset_impl());
	}
};

}

/**
 * @brief returns the offset of the value described by the structure
 */
constexpr auto offset() noexcept {
	return helpers::offset_impl();
}

/**
 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item
 * 
 * @tparam Dims: the dimension names of fixed indices
 * @param ts: parameters for fixing the indices
 */
template<char... Dims, class... Ts>
constexpr auto offset(Ts... ts) noexcept {
	return compose(fix<Dims...>(ts...), helpers::offset_impl());
}

/**
 * @brief returns the size (in bytes) of the structure
 */
struct get_size {
	using func_family = top_tag;
	constexpr get_size() noexcept = default;

	template<class T>
	constexpr auto operator()(T t) const noexcept -> decltype(t.size()) {
		return t.size();
	}
};

namespace helpers {

template<class Ptr>
struct get_at_impl : private contain<Ptr> {
	using func_family = top_tag;

	constexpr get_at_impl() noexcept = delete; // we do not want to access nondeterministic memory

	template<class T>
	explicit constexpr get_at_impl(T *ptr) noexcept : contain<Ptr>(reinterpret_cast<Ptr>(ptr)) {}

	// the return type checks whether the structure `t` is a cube and it also chooses `scalar_t<T> &` or `const scalar_t<T> &` according to constness of `Ptr` pointee
	template<class T>
	constexpr auto operator()(T t) const noexcept -> std::enable_if_t<is_cube<T>::value, std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_t<T> &, scalar_t<T> &>> {
		// accesses reference to a value with the given offset and casted to its corresponding type
		return *reinterpret_cast<std::conditional_t<std::is_const<std::remove_pointer_t<Ptr>>::value, const scalar_t<T> *, scalar_t<T> *>>(contain<Ptr>::template get<0>() + (t | offset()));
	}
};

}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 * 
 * @param ptr: the pointer to blob structure
 */
template<class V>
constexpr auto get_at(V *ptr) noexcept {
	return helpers::get_at_impl<std::conditional_t<std::is_const<V>::value, const char *, char *>>(ptr);
}

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, class V, class... Ts>
constexpr auto get_at(V *ptr, Ts... ts) noexcept {
	return compose(fix<Dims...>(ts...), get_at<V>(ptr));
}

/**
 * @brief returns the topmost dims of a structure (if the topmost structure in the substructure tree has no dims and it has only one substructure it returns the topmost dims of this substructure, recursively)
 */
struct top_dims {
	using func_family = top_tag;

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
