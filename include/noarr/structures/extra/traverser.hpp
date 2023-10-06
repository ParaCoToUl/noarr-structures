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
template<class Sig1, IsDim auto Dim, class ArgLength, class RetSig>
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
template<class Sig1, IsDim auto Dim, class... RetSigs>
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

template<class Signature, IsState State>
struct union_filter_accepted;
template<class Signature, class HeadStateItem, class... TailStateItems>
struct union_filter_accepted<Signature, state<HeadStateItem, TailStateItems...>> {
	using tail = typename union_filter_accepted<Signature, state<TailStateItems...>>::template res<>::ult;
	template<class = HeadStateItem>
	struct res {
		using ult = tail;
	};
	template<IsDim auto Dim, class ValueType> requires (Signature::template any_accept<Dim>)
	struct res<state_item<index_in<Dim>, ValueType>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};
	template<IsDim auto Dim, class ValueType> requires (Signature::template any_accept<Dim>)
	struct res<state_item<length_in<Dim>, ValueType>> {
		using ult = typename tail::template prepend<HeadStateItem>;
	};
};
template<class Signature>
struct union_filter_accepted<Signature, state<>> {
	template<class = void>
	struct res { using ult = state_items_pack<>; };
};

template<class Struct, IsState State>
using union_filter_accepted_t = typename union_filter_accepted<typename Struct::signature, State>::template res<>::ult;

} // namespace helpers

template<class... Structs>
struct union_t : contain<Structs...> {
	using contain<Structs...>::contain;

	using is = std::index_sequence_for<Structs...>;
	using signature = typename helpers::sig_union<typename to_struct<Structs>::type::signature...>::type;

	template<std::size_t Index>
	constexpr auto sub_structure() const noexcept { return this->template get<Index>(); }

private:
	template<auto Dim, std::size_t I>
	constexpr auto find_first_match() const noexcept {
		using sub_sig = typename to_struct<decltype(sub_structure<I>())>::type::signature;
		if constexpr(sub_sig::template any_accept<Dim>)
			return std::integral_constant<std::size_t, I>();
		else
			return find_first_match<Dim, I+1>();
	}
	template<IsDim auto Dim>
	static constexpr std::size_t first_match = decltype(std::declval<union_t<Structs...>>().template find_first_match<Dim, 0>())::value;
public:

	template<IsDim auto QDim>
	constexpr auto length(IsState auto state) const noexcept {
		return contain<Structs...>::template get<first_match<QDim>>().template length<QDim>(state);
	}
};

template<class ...Ts, class U = union_t<Ts...>>
constexpr U make_union(const Ts &...s) noexcept {
	return U(s...);
}

template<auto... Dim, class... IdxT> requires ((sizeof...(Dim) == sizeof...(IdxT)) && ... && IsDim<decltype(Dim)>)
constexpr auto fix(IdxT...) noexcept; // defined in setters.hpp

template<class Struct, class Order>
struct traverser_t : contain<Struct, Order> {
	using contain<Struct, Order>::contain;

	constexpr auto get_struct() const noexcept { return this->template get<0>(); }
	constexpr auto get_order() const noexcept { return this->template get<1>(); }

	[[nodiscard("returns a new traverser")]]
	constexpr auto order(IsProtoStruct auto new_order) const noexcept {
		return traverser_t<Struct, decltype(get_order() ^ new_order)>(get_struct(), get_order() ^ new_order);
	}

	template<auto... Dims, class F> requires (... && IsDim<decltype(Dims)>)
	constexpr void for_each(F f) const noexcept {
		for_sections<Dims...>([f](auto inner) constexpr noexcept { return f(inner.state()); });
	}

	// TODO add tests
	template<IsDim auto Dim, auto... Dims, class F> requires (... && IsDim<decltype(Dims)>)
	constexpr void for_sections(F f) const noexcept {
		using dim_tree = dim_tree_restrict<sig_dim_tree<typename decltype(top_struct())::signature>, dim_sequence<Dim, Dims...>>;
		static_assert((dim_tree_contains<Dim, dim_tree> && ... && dim_tree_contains<Dims, dim_tree>), "Requested dimensions are not present");
		for_each_impl(dim_tree(), f, empty_state);
	}

	// TODO add tests
	template<class F>
	constexpr void for_sections(F f) const noexcept {
		using dim_tree = sig_dim_tree<typename decltype(top_struct())::signature>;
		for_each_impl(dim_tree(), f, empty_state);
	}

	template<auto... Dims, class F> requires (... && IsDim<decltype(Dims)>)
	constexpr void for_dims(F f) const noexcept {
		using dim_tree = dim_tree_restrict<sig_dim_tree<typename decltype(top_struct())::signature>, dim_sequence<Dims...>>;
		static_assert((... && dim_tree_contains<Dims, dim_tree>), "Requested dimensions are not present");
		for_each_impl(dim_tree_from_sequence<dim_sequence<Dims...>>(), f, empty_state);
	}

	constexpr auto state() const noexcept {
		return state_at<Struct>(top_struct(), empty_state);
	}

	constexpr auto top_struct() const noexcept {
		return get_struct() ^ get_order();
	}

	template<IsDim auto Dim>
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp
	constexpr auto range() const noexcept; // defined in traverser_iter.hpp
	constexpr auto begin() const noexcept; // defined in traverser_iter.hpp
	constexpr auto end() const noexcept; // defined in traverser_iter.hpp

private:
	template<auto Dim, class Branch, class ...Branches, class F, std::size_t I, std::size_t... Is>
	constexpr void for_each_impl_dep(F f, auto state, std::index_sequence<I, Is...>) const noexcept {
		for_each_impl(Branch(), f, state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>()));
		for_each_impl_dep<Dim, Branches...>(f, state, std::index_sequence<Is...>());
	}
	template<auto Dim, class F>
	constexpr void for_each_impl_dep(F, auto, std::index_sequence<>) const noexcept {}
	template<auto Dim, class ...Branches, class F, IsState State>
	constexpr void for_each_impl(dim_tree<Dim, Branches...>, F f, State state) const noexcept {
		using dim_sig = sig_find_dim<Dim, State, typename decltype(top_struct())::signature>;
		if constexpr(dim_sig::dependent) {
			for_each_impl_dep<Dim, Branches...>(f, state, std::index_sequence_for<Branches...>());
		} else {
			std::size_t len = top_struct().template length<Dim>(state);
			for(std::size_t i = 0; i < len; i++)
				for_each_impl(Branches()..., f, state.template with<index_in<Dim>>(i));
		}
	}
	template<class F, class StateItem, class... StateItems>
	constexpr void for_each_impl(dim_sequence<>, F f, noarr::state<StateItem, StateItems...> state) const noexcept {
		f(order(fix(state)));
	}
	template<class F>
	constexpr void for_each_impl(dim_sequence<>, F f, noarr::state<>) const noexcept {
		f(*this);
	}
};

template<class... Ts>
constexpr auto traverser(const Ts &... s) noexcept
	-> traverser_t<union_t<typename to_struct<Ts>::type...>, neutral_proto>
{ return traverser(make_union(to_struct<Ts>::convert(s)...)); }

template<class... Ts>
constexpr auto traverser(union_t<Ts...> u) noexcept { return traverser_t<union_t<Ts...>, neutral_proto>(u, neutral_proto()); }

template<class T>
struct is_traverser : std::false_type {};

template<class Struct, class Order>
struct is_traverser<traverser_t<Struct, Order>> : std::true_type {};

template<class T>
struct is_traverser<const T> : is_traverser<T> {};

template<class T>
constexpr bool is_traverser_v = is_traverser<T>::value;

template<class T>
concept IsTraverser = is_traverser_v<T>;

template<IsTraverser T>
constexpr auto operator^(const T &t, IsProtoStruct auto order) noexcept {
	return t.order(order);
}

template<IsTraverser T>
constexpr auto operator|(const T &t, auto f) noexcept -> decltype(t.for_each(f)) {
	return t.for_each(f);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TRAVERSER_HPP
