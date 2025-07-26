#ifndef NOARR_STRUCTURES_TRAVERSER_HPP
#define NOARR_STRUCTURES_TRAVERSER_HPP

#include <cstddef>

#include <type_traits>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/sig_utils.hpp"
#include "../extra/to_struct.hpp"
#include "../structs/setters.hpp"

namespace noarr {

namespace helpers {

template<class Sig1, class Sig2>
struct sig_union2;

template<class Sig1, IsDim auto Dim, class ArgLength, class RetSig>
struct sig_union2<Sig1, function_sig<Dim, ArgLength, RetSig>> {
	using ret = typename sig_union2<Sig1, RetSig>::type;
	template<bool = Sig1::template any_accept<Dim>, class = void>
	struct ty;

	template<class Useless>
	struct ty<true, Useless> {
		using pe = ret;
	}; // TODO ArgLength union

	template<class Useless>
	struct ty<false, Useless> {
		using pe = function_sig<Dim, ArgLength, ret>;
	};

	using type = typename ty<>::pe;
};

template<class Sig1, IsDim auto Dim, class... RetSigs>
struct sig_union2<Sig1, dep_function_sig<Dim, RetSigs...>> {
	static_assert(value_always_false<Dim>, "Unsupported");
};

template<class Sig1, class ValueType>
struct sig_union2<Sig1, scalar_sig<ValueType>> {
	using type = Sig1;
};

template<class... Sigs>
struct sig_union;

template<class Sig1, class Sig2, class... Sigs>
struct sig_union<Sig1, Sig2, Sigs...> {
	using type = typename sig_union<typename sig_union2<Sig1, Sig2>::type, Sigs...>::type;
};

template<class Sig>
struct sig_union<Sig> {
	using type = Sig;
};

template<class Signature, IsState State>
struct union_filter_accepted;

template<class Signature, class HeadStateItem, class... TailStateItems>
struct union_filter_accepted<Signature, state<HeadStateItem, TailStateItems...>> {
	using tail = typename union_filter_accepted<Signature, state<TailStateItems...>>::template res<>::ult;

	template<class = HeadStateItem>
	struct res {
		using ult = tail;
	};

	template<IsDim auto Dim, class ValueType>
	requires (Signature::template any_accept<Dim>)
	struct res<state_item<index_in<Dim>, ValueType>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};

	template<IsDim auto Dim, class ValueType>
	requires (Signature::template any_accept<Dim>)
	struct res<state_item<length_in<Dim>, ValueType>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};
};

template<class Signature>
struct union_filter_accepted<Signature, state<>> {
	template<class = void>
	struct res {
		using ult = state_items_pack<>;
	};
};

template<class Struct, IsState State>
using union_filter_accepted_t = typename union_filter_accepted<typename Struct::signature, State>::template res<>::ult;

} // namespace helpers

template<class... Structs>
struct union_t : strict_contain<Structs...> {
	using base = strict_contain<Structs...>;
	using base::base;

	using is = std::index_sequence_for<Structs...>;
	using signature = typename helpers::sig_union<typename to_struct<Structs>::type::signature...>::type;

	template<std::size_t Index>
	[[nodiscard("returns a copy of the underlying struct")]]
	constexpr auto sub_structure() const noexcept {
		return this->template get<Index>();
	}

	template<std::size_t Index>
	using sub_structure_t = std::remove_cvref_t<decltype(std::declval<base>().template get<Index>())>;

private:
	template<auto Dim, std::size_t I>
	[[nodiscard("returns the index of the first struct that accepts the dimension")]]
	static constexpr std::size_t find_first_match() noexcept {
		using sub_sig = typename to_struct<sub_structure_t<I>>::type::signature;
		if constexpr (sub_sig::template any_accept<Dim>) {
			return I;
		} else if constexpr (I < sizeof...(Structs) - 1) {
			return find_first_match<Dim, I + 1>();
		} else {
			return sizeof...(Structs);
		}
	}

	template<auto Dim>
	static constexpr std::size_t first_match = find_first_match<Dim, 0>();

public:
	template<auto Dim, IsState State>
	requires IsDim<decltype(Dim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (first_match<Dim> < sizeof...(Structs)) {
			return sub_structure_t<first_match<Dim>>::template has_length<Dim, State>();
		} else {
			return false;
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard("returns the length of the first struct that accepts the dimension")]]
	constexpr auto length(State state) const noexcept {
		return sub_structure<first_match<QDim>>().template length<QDim>(state);
	}
};

template<class... Ts, class U = union_t<Ts...>>
constexpr U make_union(const Ts &...s) noexcept {
	return U(s...);
}

template<class Struct, class Order>
struct traverser_t : strict_contain<Struct, Order> {
	using strict_contain<Struct, Order>::strict_contain;

	[[nodiscard("returns a copy of the underlying struct")]]
	constexpr auto get_struct() const noexcept {
		return this->template get<0>();
	}

	[[nodiscard("returns a copy of the underlying order")]]
	constexpr auto get_order() const noexcept {
		return this->template get<1>();
	}

	template<IsProtoStruct NewOrder>
	[[nodiscard("returns a new traverser")]]
	constexpr auto order(NewOrder new_order) const noexcept {
		return traverser_t<Struct, decltype(get_order() ^ new_order)>(get_struct(), get_order() ^ new_order);
	}

	template<auto... Dims, class F>
	requires IsDimPack<decltype(Dims)...>
	constexpr void for_each(F f) const {
		for_sections<Dims...>([f](auto inner) constexpr { f(inner.state()); });
	}

	template<auto Dim, auto... Dims, class F>
	requires IsDimPack<decltype(Dim), decltype(Dims)...>
	constexpr void for_sections(F f) const {
		using dim_tree_helper =
			dim_tree_filter<sig_dim_tree<typename decltype(top_struct())::signature>, in_dim_sequence<Dim, Dims...>>;
		static_assert((dim_tree_contains<Dim, dim_tree_helper> && ... && dim_tree_contains<Dims, dim_tree_helper>),
		              "Requested dimensions are not present");
		for_each_impl(dim_tree_helper(), f, empty_state);
	}

	template<class F>
	constexpr void for_sections(F f) const {
		using dim_tree_helper = sig_dim_tree<typename decltype(top_struct())::signature>;
		for_each_impl(dim_tree_helper(), f, empty_state);
	}

	template<auto... Dims, class F>
	requires IsDimPack<decltype(Dims)...>
	constexpr void for_dims(F f) const {
		using dim_tree_helper =
			dim_tree_filter<sig_dim_tree<typename decltype(top_struct())::signature>, in_dim_sequence<Dims...>>;
		static_assert((... && dim_tree_contains<Dims, dim_tree_helper>), "Requested dimensions are not present");
		using reordered_tree = dim_tree_reorder<dim_tree_helper, Dims...>;
		for_each_impl(reordered_tree(), f, empty_state);
	}

	[[nodiscard("construct an object representing the state of the traverser")]]
	constexpr auto state() const noexcept {
		return state_at<Struct>(top_struct(), empty_state);
	}

	[[nodiscard("returns a copy of the top struct (combination of the underlying struct and order)")]]
	constexpr auto top_struct() const noexcept {
		return get_struct() ^ get_order();
	}

	template<auto Dim>
	requires IsDim<decltype(Dim)>
	[[nodiscard]]
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp

	[[nodiscard]]
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp

	[[nodiscard]]
	constexpr auto begin() const noexcept; // defined in traverser_iter.hpp

	[[nodiscard]]
	constexpr auto end() const noexcept; // defined in traverser_iter.hpp

private:
	template<auto Dim, class Branch, class... Branches, class F, std::size_t I, std::size_t... Is>
	constexpr void for_each_impl_dep(F f, auto state, std::index_sequence<I, Is...> /*is*/) const {
		for_each_impl(Branch(), f, state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>()));
		for_each_impl_dep<Dim, Branches...>(f, state, std::index_sequence<Is...>());
	}

	// the bottom case
	template<auto Dim, class F>
	requires IsDim<decltype(Dim)>
	constexpr void for_each_impl_dep(F /*f*/, auto /*state*/, std::index_sequence<> /*is*/) const {}

	template<auto Dim, class... Branches, class F, class State>
	requires IsState<State> && IsDim<decltype(Dim)>
	constexpr void for_each_impl(dim_tree<Dim, Branches...> /*dt*/, F f, State state) const {
		using dim_sig = sig_find_dim<Dim, State, typename decltype(top_struct())::signature>;
		if constexpr (dim_sig::dependent) {
			for_each_impl_dep<Dim, Branches...>(f, state, std::index_sequence_for<Branches...>());
		} else {
			const std::size_t len = top_struct().template length<Dim>(state);
			for (std::size_t i = 0; i < len; i++) {
				for_each_impl(Branches()..., f, state.template with<index_in<Dim>>(i));
			}
		}
	}

	template<class F, class StateItem, class... StateItems>
	constexpr void for_each_impl(dim_sequence<> /*ds*/, F f, noarr::state<StateItem, StateItems...> state) const {
		f(order(fix(state)));
	}

	template<class F>
	constexpr void for_each_impl(dim_sequence<> /*ds*/, F f, noarr::state<> /*state*/) const {
		f(*this);
	}

	template<class DimTree, class F, IsState State>
	friend constexpr void for_each_impl(const traverser_t &traverser, DimTree /*dt*/, F f, State state) {
		traverser.for_each_impl(DimTree(), f, state);
	}
};

template<class... Ts>
constexpr auto traverser(const Ts &...s) noexcept
	-> traverser_t<union_t<typename to_struct<Ts>::type...>, neutral_proto> {
	return traverser(make_union(to_struct<Ts>::convert(s)...));
}

template<class... Ts>
constexpr auto traverser(union_t<Ts...> u) noexcept {
	return traverser_t<union_t<Ts...>, neutral_proto>(u, neutral_proto());
}

template<class T>
concept IsTraverser = IsSpecialization<T, traverser_t>;

template<class T>
struct to_traverser : std::false_type {};

template<class T>
constexpr bool to_traverser_v = to_traverser<T>::value;

template<class T>
using to_traverser_t = typename to_traverser<T>::type;

template<class T>
concept ToTraverser = to_traverser_v<std::remove_cvref_t<T>>;

template<class T>
requires ToTraverser<T>
constexpr decltype(auto) convert_to_traverser(T &&t) noexcept {
	return to_traverser<std::remove_cvref_t<T>>::convert(std::forward<T>(t));
}

template<class Struct, class Order>
struct to_traverser<traverser_t<Struct, Order>> : std::true_type {
	using type = traverser_t<Struct, Order>;

	[[nodiscard]]
	static constexpr type convert(const traverser_t<Struct, Order> &traverser) noexcept {
		return traverser;
	}
};

template<IsTraverser T, IsProtoStruct Order>
[[nodiscard("returns a new traverser")]]
constexpr auto operator^(const T &t, Order order) noexcept {
	return t.order(order);
}

template<IsTraverser T>
struct to_state<T> : std::true_type {
	using type = std::remove_cvref_t<decltype(std::declval<T>().state())>;

	[[nodiscard]]
	static constexpr type convert(const T &t) noexcept {
		return t.state();
	}
};

namespace helpers {

template<class F, auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct for_each_t : public F {};

template<class F, auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct for_dims_t : public F {};

template<class F, auto... Dims>
requires IsDimPack<decltype(Dims)...>
struct for_sections_t : public F {};

} // namespace helpers

template<auto... Dims, class F>
requires IsDimPack<decltype(Dims)...>
constexpr auto for_each(F &&f) noexcept {
	return helpers::for_each_t<std::remove_cvref_t<F>, Dims...>{std::forward<F>(f)};
}

template<auto... Dims, class F>
requires IsDimPack<decltype(Dims)...>
constexpr auto for_dims(F &&f) noexcept {
	return helpers::for_dims_t<std::remove_cvref_t<F>, Dims...>{std::forward<F>(f)};
}

template<auto... Dims, class F>
requires IsDimPack<decltype(Dims)...>
constexpr auto for_sections(F &&f) noexcept {
	return helpers::for_sections_t<std::remove_cvref_t<F>, Dims...>{std::forward<F>(f)};
}

template<IsTraverser T>
constexpr auto operator|(const T &t, auto f) -> decltype(t.for_each(f)) {
	return t.for_each(f);
}

template<IsTraverser T, auto... Dims, class F>
constexpr auto operator|(const T &t, const helpers::for_each_t<F, Dims...> &f)
	-> decltype(t.template for_each<Dims...>(f)) {
	return t.template for_each<Dims...>(f);
}

template<IsTraverser T, auto... Dims, class F>
constexpr auto operator|(const T &t, const helpers::for_dims_t<F, Dims...> &f)
	-> decltype(t.template for_dims<Dims...>(f)) {
	return t.template for_dims<Dims...>(f);
}

template<IsTraverser T, auto... Dims, class F>
constexpr auto operator|(const T &t, const helpers::for_sections_t<F, Dims...> &f)
	-> decltype(t.template for_sections<Dims...>(f)) {
	return t.template for_sections<Dims...>(f);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TRAVERSER_HPP
