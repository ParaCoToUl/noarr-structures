#ifndef NOARR_STRUCTURES_UTILITY_HPP
#define NOARR_STRUCTURES_UTILITY_HPP

#include <cstddef>

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

namespace noarr {

template<auto Tag>
requires std::is_empty_v<decltype(Tag)> || requires { Tag == Tag; }
struct dim {
	/* Currently unspecified content */
	using tag_type = decltype(Tag);
	static constexpr tag_type tag = Tag;

	template<auto Tag2>
	requires std::same_as<decltype(Tag), decltype(Tag2)>
	constexpr bool operator==(dim<Tag2> /*other*/) const noexcept {
		if constexpr (std::is_empty_v<decltype(Tag2)>) {
			return true;
		} else {
			return Tag == Tag2;
		}
	}

	template<auto Tag2>
	constexpr bool operator==(dim<Tag2> /*other*/) const noexcept {
		return false;
	}

	friend constexpr bool operator==(char /*lhs*/, dim /*rhs*/) noexcept { return false; }

	friend constexpr bool operator==(dim /*lhs*/, char /*rhs*/) noexcept { return false; }
};

template<class T>
struct is_dim_impl : std::false_type {};

template<>
struct is_dim_impl<char> : std::true_type {};

template<auto Tag>
struct is_dim_impl<dim<Tag>> : std::true_type {};

template<class T>
using is_dim = is_dim_impl<std::remove_cvref_t<T>>;

template<class T>
static constexpr bool is_dim_v = is_dim<T>::value;

template<class T>
concept IsDim = is_dim_v<std::remove_cvref_t<T>>;

template<class... Ts>
concept IsDimPack = (... && IsDim<Ts>);

template<auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct dim_sequence {
	using type = dim_sequence;
	using size_type = std::size_t;

	static constexpr size_type size = sizeof...(Dims);

	template<auto Dim>
	requires IsDim<decltype(Dim)>
	static constexpr bool contains = (... || (Dim == Dims));

	template<auto Dim>
	requires IsDim<decltype(Dim)>
	using push_back = dim_sequence<Dims..., Dim>;

	template<auto Dim>
	requires IsDim<decltype(Dim)>
	using push_front = dim_sequence<Dim, Dims...>;
};

template<class T>
struct is_dim_sequence : std::false_type {};

template<auto... Dims>
struct is_dim_sequence<dim_sequence<Dims...>> : std::true_type {};

template<class T>
static constexpr bool is_dim_sequence_v = is_dim_sequence<T>::value;

template<class T>
concept IsDimSequence = is_dim_sequence_v<std::remove_cvref_t<T>>;

template<class T>
struct dim_sequence_contains;

template<auto... Dims>
struct dim_sequence_contains<dim_sequence<Dims...>> {
	template<auto Dim>
	requires IsDim<decltype(Dim)>
	static constexpr bool value = (... || (Dim == Dims));
};

struct dim_accepter {
	template<auto Dim>
	requires IsDim<decltype(Dim)>
	static constexpr bool value = true;
};

struct dim_identity_mapper {
	template<auto Dim>
	requires IsDim<decltype(Dim)>
	static constexpr auto value = Dim;
};

template<auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct in_dim_sequence {
	template<auto Dim>
	requires IsDim<decltype(Dim)>
	using value_type = bool;

	template<auto Dim>
	requires IsDim<decltype(Dim)>
	static constexpr bool value = (... || (Dim == Dims));
};

namespace helpers {

template<class... Packs>
struct integer_sequence_concat_impl;

template<class T, T... vs1, T... vs2, class... Packs>
struct integer_sequence_concat_impl<std::integer_sequence<T, vs1...>, std::integer_sequence<T, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_impl<std::integer_sequence<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T... vs1>
struct integer_sequence_concat_impl<std::integer_sequence<T, vs1...>> {
	using type = std::integer_sequence<T, vs1...>;
};

template<class Sep, class... Packs>
struct integer_sequence_concat_sep_impl;

template<class T, T... vs1, T... vs2, T... sep, class... Packs>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, vs1...>,
                                        std::integer_sequence<T, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>,
	                                                       std::integer_sequence<T, vs1..., vs2...>, Packs...>::type;
};

template<class T, T v1, T v2, T... vs1, T... vs2, T... sep, class... Packs>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, v1, vs1...>,
                                        std::integer_sequence<T, v2, vs2...>, Packs...> {
	using type = typename integer_sequence_concat_sep_impl<
		std::integer_sequence<T, sep...>, std::integer_sequence<T, v1, vs1..., sep..., v2, vs2...>, Packs...>::type;
};

template<class T, T... vs1, T... sep>
struct integer_sequence_concat_sep_impl<std::integer_sequence<T, sep...>, std::integer_sequence<T, vs1...>> {
	using type = std::integer_sequence<T, vs1...>;
};

template<class... Packs>
struct dim_sequence_concat_impl;

template<auto... vs1, auto... vs2, class... Packs>
struct dim_sequence_concat_impl<dim_sequence<vs1...>, dim_sequence<vs2...>, Packs...> {
	using type = typename dim_sequence_concat_impl<dim_sequence<vs1..., vs2...>, Packs...>::type;
};

template<auto... vs1>
struct dim_sequence_concat_impl<dim_sequence<vs1...>> {
	using type = dim_sequence<vs1...>;
};

template<class In, class Pred>
struct dim_sequence_filter_impl;

template<auto v, auto... vs, class Pred>
requires (Pred::template value<v>)
struct dim_sequence_filter_impl<dim_sequence<v, vs...>, Pred> {
	using type =
		typename dim_sequence_concat_impl<dim_sequence<v>,
	                                      typename dim_sequence_filter_impl<dim_sequence<vs...>, Pred>::type>::type;
};

template<auto v, auto... vs, class Pred>
struct dim_sequence_filter_impl<dim_sequence<v, vs...>, Pred> {
	using type = typename dim_sequence_filter_impl<dim_sequence<vs...>, Pred>::type;
};

template<class Pred>
struct dim_sequence_filter_impl<dim_sequence<>, Pred> {
	using type = dim_sequence<>;
};

template<auto v, class... Branches>
struct dim_tree {
	static constexpr auto root_value = v;

	template<std::size_t I>
	requires (I < sizeof...(Branches))
	using branch = std::tuple_element_t<I, std::tuple<Branches...>>;
};

template<auto v, class Tree>
struct dim_tree_contains_impl : std::false_type {};

template<auto v, class Tree>
static constexpr bool dim_tree_contains = dim_tree_contains_impl<v, Tree>::value;

template<auto v, auto V, class... Branches>
requires (v != V)
struct dim_tree_contains_impl<v, dim_tree<V, Branches...>>
	: std::bool_constant<(... || dim_tree_contains<v, Branches>)> {};

template<auto v, class... Branches>
struct dim_tree_contains_impl<v, dim_tree<v, Branches...>> : std::true_type {};

template<auto v, class Tree>
struct dim_tree_branches_impl : std::integral_constant<std::size_t, 0> {};

template<auto v, class Tree>
static constexpr auto dim_tree_branches = dim_tree_branches_impl<v, Tree>::value;

template<std::size_t... Is>
struct max_impl : std::integral_constant<std::size_t, 0> {};

template<std::size_t I>
struct max_impl<I> : std::integral_constant<std::size_t, I> {};

template<std::size_t I, std::size_t J>
struct max_impl<I, J> : std::integral_constant<std::size_t, (I > J ? I : J)> {};

template<std::size_t I, std::size_t... Is>
struct max_impl<I, Is...> : max_impl<I, max_impl<Is...>::value> {};

template<auto v, auto V, class... Branches>
requires (v != V)
struct dim_tree_branches_impl<v, dim_tree<V, Branches...>> {
	static constexpr auto value = max_impl<dim_tree_branches_impl<v, Branches>::value...>::value;
};

template<auto v, class... Branches>
struct dim_tree_branches_impl<v, dim_tree<v, Branches...>> {
	static constexpr auto value = sizeof...(Branches);
};

template<auto v, std::size_t I, class Tree>
struct dim_tree_fix_impl {
	using type = Tree;
};

template<auto v, std::size_t I, class Tree>
using dim_tree_fix = typename dim_tree_fix_impl<v, I, Tree>::type;

template<auto v, std::size_t I, auto V, class... Branches>
requires (v != V)
struct dim_tree_fix_impl<v, I, dim_tree<V, Branches...>> {
	using type = dim_tree<V, typename dim_tree_fix_impl<v, I, Branches>::type...>;
};

template<auto v, std::size_t I, class... Branches>
struct dim_tree_fix_impl<v, I, dim_tree<v, Branches...>> {
	using type = std::tuple_element_t<I, std::tuple<Branches...>>;
};

template<auto v, class Tree>
struct dim_tree_hoist_impl {
private:
	static constexpr auto branches = dim_tree_branches_impl<v, Tree>::value;
	using indices = std::make_index_sequence<branches>;
	using tree = decltype([]<std::size_t... Is>(std::index_sequence<Is...>) {
		return dim_tree<v, typename dim_tree_fix_impl<v, Is, Tree>::type...>{};
	}(indices{}));

public:
	using type = tree;
};

template<class Tree, auto... vs>
struct dim_tree_reorder_impl;

template<class Tree, auto... vs>
using dim_tree_reorder = typename dim_tree_reorder_impl<Tree, vs...>::type;

template<class Tree>
struct dim_tree_reorder_impl<Tree> {
	using type = Tree;
};

template<auto v, auto... vs, class Tree>
struct dim_tree_reorder_impl<Tree, v, vs...> {
	using type = typename dim_tree_hoist_impl<v, typename dim_tree_reorder_impl<Tree, vs...>::type>::type;
};

template<class Tree, class Pred>
struct dim_tree_filter_impl;

template<auto v, class Branch, class Pred>
requires (!Pred::template value<v>)
struct dim_tree_filter_impl<dim_tree<v, Branch>, Pred> {
	using type = typename dim_tree_filter_impl<Branch, Pred>::type;
};

template<auto v, class... Branches, class Pred>
requires (Pred::template value<v>)
struct dim_tree_filter_impl<dim_tree<v, Branches...>, Pred> {
	using type = dim_tree<v, typename dim_tree_filter_impl<Branches, Pred>::type...>;
};

template<class Pred>
struct dim_tree_filter_impl<dim_sequence<>, Pred> {
	using type = dim_sequence<>;
};

template<class Sequence>
struct dim_tree_from_sequence_impl;

template<auto v, auto... vs>
struct dim_tree_from_sequence_impl<dim_sequence<v, vs...>> {
	using type = dim_tree<v, typename dim_tree_from_sequence_impl<dim_sequence<vs...>>::type>;
};

template<>
struct dim_tree_from_sequence_impl<dim_sequence<>> {
	using type = dim_sequence<>;
};

template<class DimTree>
struct dim_tree_to_sequence_impl;

template<auto v, class... Branches>
requires (sizeof...(Branches) == 1)
struct dim_tree_to_sequence_impl<dim_tree<v, Branches...>> {
	using type =
		typename dim_sequence_concat_impl<dim_sequence<v>, typename dim_tree_to_sequence_impl<Branches...>::type>::type;
};

template<>
struct dim_tree_to_sequence_impl<dim_sequence<>> {
	using type = dim_sequence<>;
};

} // namespace helpers

/**
 * @brief concatenates multiple integral `Packs`
 *
 * @tparam Packs: the input integral packs
 */
template<class... Packs>
using integer_sequence_concat = typename helpers::integer_sequence_concat_impl<Packs...>::type;

/**
 * @brief concatenates multiple integral packs (the 2nd, 3rd etc. member of `Packs`) pasting the 1st member of `Packs`
 * between each consecutive packs
 *
 * @tparam Packs: the input integral packs, the first one is the separator used when concatenating
 */
template<class... Packs>
using integer_sequence_concat_sep = typename helpers::integer_sequence_concat_sep_impl<Packs...>::type;

template<class In, class Pred>
using dim_sequence_filter = typename helpers::dim_sequence_filter_impl<In, Pred>::type;

using helpers::dim_tree;

template<class In, class Pred>
using dim_tree_filter = typename helpers::dim_tree_filter_impl<In, Pred>::type;

using helpers::dim_tree_contains;

template<class In, auto... Dims>
requires IsDimPack<decltype(Dims)...>
using dim_tree_reorder = typename helpers::dim_tree_reorder_impl<In, Dims...>::type;

template<class Seq>
using dim_tree_from_sequence = typename helpers::dim_tree_from_sequence_impl<Seq>::type;

template<class DimTree>
using dim_tree_to_sequence = typename helpers::dim_tree_to_sequence_impl<DimTree>::type;

template<std::size_t I>
struct lit_t : std::integral_constant<std::size_t, I> {
	constexpr auto operator()() const =
		delete; // using `lit<42>()` by mistake should be rejected, not evaluate to dynamic size_t of 42
};

template<std::size_t I>
constexpr lit_t<I> lit;

struct empty_t {};

template<class T>
struct dim_pred_not {
	template<auto Dim>
	using value_type = bool;

	template<auto Dim>
	static constexpr bool value = !T::template value<Dim>;
};

template<class T, class U>
struct dim_pred_or {
	template<auto Dim>
	using value_type = bool;

	template<auto Dim>
	static constexpr bool value = T::template value<Dim> || U::template value<Dim>;
};

template<class T, class U>
struct dim_pred_and {
	template<auto Dim>
	using value_type = bool;

	template<auto Dim>
	static constexpr bool value = T::template value<Dim> && U::template value<Dim>;
};

template<class>
static constexpr bool always_false = false;
template<auto>
static constexpr bool value_always_false = false;

template<class T>
struct some {
	static constexpr bool present = true;
	T value;

	template<class F>
	[[nodiscard]]
	constexpr auto and_then(const F &f) const noexcept -> some<decltype(f(value))> {
		return {f(value)};
	}
};

struct none {
	static constexpr bool present = false;

	template<class F>
	[[nodiscard]]
	constexpr none and_then(const F & /*f*/) const noexcept {
		return {};
	}
};

template<class T>
concept IsSimple =
	std::is_standard_layout_v<T> &&
	(!std::is_empty_v<T> || std::is_trivially_default_constructible_v<T>)/* empty -> trivially_default_constructible */
	&&(!std::is_default_constructible_v<T> ||
       std::is_trivially_default_constructible_v<T>)/* default_constructible -> trivially_default_constructible */
	&&std::is_trivially_copy_constructible_v<T> &&
	std::is_trivially_move_constructible_v<T> &&
	(!std::is_copy_assignable_v<T> ||
     std::is_trivially_copy_assignable_v<T>)/* copy_assignable -> trivially_copy_assignable */
	&&(!std::is_move_assignable_v<T> ||
       std::is_trivially_move_assignable_v<T>)/* move_assignable -> trivially_move_assignable */
	&&std::is_trivially_destructible_v<T>;

template<class T>
concept IsContainable =
	(!std::is_empty_v<T> || std::is_default_constructible_v<T>); /* empty -> default_constructible */

template<class T, template<class...> class Template>
struct is_specialization : std::false_type {};

template<template<class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

template<class T, template<class...> class Template>
static constexpr bool is_specialization_v = is_specialization<T, Template>::value;

template<class T, template<class...> class Template>
concept IsSpecialization = is_specialization_v<std::remove_cvref_t<T>, Template>;

namespace constexpr_arithmetic {

template<std::size_t N>
using make_const = std::integral_constant<std::size_t, N>;

template<std::size_t A, std::size_t B>
constexpr make_const<A + B> operator+(make_const<A> /*lhs*/, make_const<B> /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr T operator+(make_const<0> /*lhs*/, T rhs) noexcept {
	return rhs;
}

template<std::integral T>
constexpr T operator+(T lhs, make_const<0> /*rhs*/) noexcept {
	return lhs;
}

template<std::size_t A, std::size_t B>
constexpr make_const<A - B> operator-(make_const<A> /*lhs*/, make_const<B> /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr T operator-(make_const<0> /*lhs*/, T rhs) noexcept {
	return -rhs;
}

template<std::integral T>
constexpr T operator-(T lhs, make_const<0> /*rhs*/) noexcept {
	return lhs;
}

template<std::size_t A, std::size_t B>
constexpr make_const<A * B> operator*(make_const<A> /*lhs*/, make_const<B> /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr make_const<0> operator*(make_const<0> /*lhs*/, T /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr make_const<0> operator*(T /*lhs*/, make_const<0> /*rhs*/) noexcept {
	return {};
}

template<std::size_t A, std::size_t B>
constexpr make_const<A / B> operator/(make_const<A> /*lhs*/, make_const<B> /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr make_const<0> operator/(make_const<0> /*lhs*/, T /*rhs*/) noexcept {
	return {};
}

template<std::size_t A, std::size_t B>
constexpr make_const<A % B> operator%(make_const<A> /*lhs*/, make_const<B> /*rhs*/) noexcept {
	return {};
}

template<std::integral T>
constexpr make_const<0> operator%(make_const<0> /*lhs*/, T /*rhs*/) noexcept {
	return {};
}

} // namespace constexpr_arithmetic

} // namespace noarr

#endif // NOARR_STRUCTURES_UTILITY_HPP
