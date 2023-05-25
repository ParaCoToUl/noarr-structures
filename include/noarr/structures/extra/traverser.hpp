#ifndef NOARR_STRUCTURES_TRAVERSER_HPP
#define NOARR_STRUCTURES_TRAVERSER_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/sig_utils.hpp"
#include "../extra/to_struct.hpp"

namespace noarr {

namespace helpers {

template<class Sig1, class Sig2>
struct sig_union2;
template<class Sig1, char Dim, class ArgLength, class RetSig>
struct sig_union2<Sig1, function_sig<Dim, ArgLength, RetSig>> {
	using ret = typename sig_union2<Sig1, RetSig>::type;
	template<bool = Sig1::template any_accept<Dim>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = ret; }; // TODO ArgLength union
	template<class Useless>
	struct ty<false, Useless> { using pe = function_sig<Dim, ArgLength, ret>; };
	using type = typename ty<>::pe;
};
template<class Sig1, char Dim, class... RetSigs>
struct sig_union2<Sig1, dep_function_sig<Dim, RetSigs...>> {
	// TODO
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

template<class Signature, class State>
struct union_filter_accepted;
template<class Signature, class HeadStateItem, class... TailStateItems>
struct union_filter_accepted<Signature, state<HeadStateItem, TailStateItems...>> {
	using tail = typename union_filter_accepted<Signature, state<TailStateItems...>>::template res<>::ult;
	template<class = HeadStateItem, class = void>
	struct res {
		using ult = tail;
	};
	template<char Dim, class ValueType>
	struct res<state_item<index_in<Dim>, ValueType>, std::enable_if_t<Signature::template any_accept<Dim>>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};
	template<char Dim, class ValueType>
	struct res<state_item<length_in<Dim>, ValueType>, std::enable_if_t<Signature::template any_accept<Dim>>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};
};
template<class Signature>
struct union_filter_accepted<Signature, state<>> {
	template<class = void>
	struct res { using ult = state_items_pack<>; };
};

template<class Struct, class State>
using union_filter_accepted_t = typename union_filter_accepted<typename Struct::signature, State>::template res<>::ult;

} // namespace helpers

template<class... Structs>
struct union_t : contain<Structs...> {
	using base = contain<Structs...>;
	using base::base;

	using is = std::index_sequence_for<Structs...>;
	using signature = typename helpers::sig_union<typename Structs::signature...>::type;

	template<std::size_t Index>
	constexpr auto sub_structure() const noexcept { return base::template get<Index>(); }

private:
	template<char Dim, std::size_t I>
	constexpr auto find_first_match() {
		using sub_t = decltype(sub_structure<I>());
		using sub_sig = typename sub_t::signature;
		if constexpr(sub_sig::template any_accept<Dim>)
			return std::integral_constant<std::size_t, I>();
		else
			return find_first_match<Dim, I+1>();
	}
	template<char Dim>
	static constexpr std::size_t first_match = decltype(std::declval<union_t<Structs...>>().template find_first_match<Dim, 0>())::value;
public:

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		return base::template get<first_match<QDim>>().template length<QDim>(state);
	}
};

template<class ...Ts, class U = union_t<typename to_struct<Ts>::type...>>
constexpr U make_union(const Ts &...s) noexcept {
	return U(to_struct<Ts>::convert(s)...);
}

template<char... Dim, class... IdxT>
constexpr auto fix(IdxT...) noexcept; // defined in setters.hpp

template<class Struct, class Order>
struct traverser_t : contain<Struct, Order> {
	using base = contain<Struct, Order>;
	using base::base;

	constexpr auto get_struct() const noexcept { return base::template get<0>(); }
	constexpr auto get_order() const noexcept { return base::template get<1>(); }

	template<class NewOrder>
	constexpr auto order(NewOrder new_order) const noexcept {
		return traverser_t<Struct, decltype(get_order() ^ new_order)>(get_struct(), get_order() ^ new_order);
	}

	template<char Dim, char... Dims, class F>
	constexpr void for_each(F f) const noexcept {
		for_sections<Dim, Dims...>([f](auto inner) { return f(inner.state()); });
	}

	template<class F>
	constexpr void for_each(F f) const noexcept {
		for_sections([f](auto inner) { return f(inner.state()); });
	}

	// TODO add tests
	template<char Dim, char... Dims, class F>
	constexpr void for_sections(F f) const noexcept {
		using dim_tree = sig_dim_tree<typename decltype(top_struct())::signature>;
		static_assert((integer_tree_contains<char, Dim, dim_tree> && ... && integer_tree_contains<char, Dims, dim_tree>), "Requested dimensions are not present");
		for_each_impl(integer_tree_restrict<dim_tree, char_sequence<Dim, Dims...>>(), f, empty_state);
	}

	// TODO add tests
	template<class F>
	constexpr void for_sections(F f) const noexcept {
		using dim_tree = sig_dim_tree<typename decltype(top_struct())::signature>;
		for_each_impl(dim_tree(), f, empty_state);
	}


	template<char... Dims, class F>
	constexpr void for_dims(F f) const noexcept {
		using dim_tree = sig_dim_tree<typename decltype(top_struct())::signature>;
		static_assert((... && integer_tree_contains<char, Dims, dim_tree>), "Requested dimensions are not present");
		for_each_impl(integer_tree_from_sequence<char_sequence<Dims...>>(), f, empty_state);
	}

	constexpr auto state() const noexcept {
		return state_at<Struct>(top_struct(), empty_state);
	}

	constexpr auto top_struct() const noexcept {
		return get_struct() ^ get_order();
	}

	template<char Dim>
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp
	constexpr auto begin() const noexcept; // defined in traverser_iter.hpp
	constexpr auto end() const noexcept; // defined in traverser_iter.hpp

private:
	template<char Dim, class ...Branches, class F, class State, std::size_t... I>
	constexpr void for_each_impl_dep(F f, State state, std::index_sequence<I...>) const noexcept {
		if constexpr (sizeof...(Branches) == 1) {
			(..., for_each_impl(Branches()..., f, state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>())));
		} else {
			(..., for_each_impl(Branches(), f, state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>())));
		}
	}
	template<char Dim, class ...Branches, class F, class State>
	constexpr void for_each_impl(integer_tree<char, Dim, Branches...>, F f, State state) const noexcept {
		using dim_sig = sig_find_dim<Dim, State, typename decltype(top_struct())::signature>;
		if constexpr(dim_sig::dependent) {
			constexpr std::size_t len = std::tuple_size_v<typename dim_sig::ret_sig_tuple>;
			for_each_impl_dep<Dim, Branches...>(f, state, std::make_index_sequence<len>());
		} else {
			std::size_t len = top_struct().template length<Dim>(state);
			for(std::size_t i = 0; i < len; i++)
				for_each_impl(Branches()..., f, state.template with<index_in<Dim>>(i));
		}
	}
	template<class F, char... Dims, class... IdxT>
	constexpr void for_each_impl(char_sequence<>, F f, noarr::state<state_item<index_in<Dims>, IdxT>...> state) const noexcept {
		f(order((... ^ fix<Dims>(state.template get<index_in<Dims>>()))));
	}
	template<class F>
	constexpr void for_each_impl(char_sequence<>, F f, noarr::state<>) const noexcept {
		f(*this);
	}
};

template<class... Ts>
constexpr auto traverser(const Ts &... s) noexcept 
	-> traverser_t<union_t<typename to_struct<Ts>::type...>, neutral_proto>
{ return traverser(make_union(to_struct<Ts>::convert(s)...)); }

template<class... Ts>
constexpr auto traverser(union_t<Ts...> u) noexcept { return traverser_t<union_t<Ts...>, neutral_proto>(u, neutral_proto()); }

} // namespace noarr

#endif // NOARR_STRUCTURES_TRAVERSER_HPP
