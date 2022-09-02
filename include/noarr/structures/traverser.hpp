#ifndef NOARR_STRUCTURES_TRAVERSER_HPP
#define NOARR_STRUCTURES_TRAVERSER_HPP

#include "funcs.hpp"

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
	static_assert(always_false_dim<Dim>, "Unsupported");
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
		if constexpr(decltype(sub_structure<I>())::signature::template any_accept<Dim>)
			return std::integral_constant<std::size_t, I>();
		else
			return find_first_match<Dim, I+1>();
	}
	template<char Dim>
	static constexpr std::size_t first_match = decltype(std::declval<union_t<Structs...>>().template find_first_match<Dim, 0>())::value;
public:

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		return base::template get<first_match<QDim>>().template length<QDim>(state);
	}

	template<class F, class State>
	static constexpr void single_iter(F f, State state) noexcept {
		f(state.restrict(typename helpers::union_filter_accepted<typename Structs::signature, State>::template res<>::ult())...);
	}
};

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

	template<class F>
	constexpr void for_each(F f) {
		auto top_struct = get_struct() ^ get_order();
		for_each_impl<typename decltype(top_struct)::signature>::for_each(top_struct, f, state<>());
	}

private:
	template<class T>
	struct for_each_impl : std::false_type {};
	template<char Dim, class ArgLength, class RetSig>
	struct for_each_impl<function_sig<Dim, ArgLength, RetSig>> {
		template<class TopStruct, class F, class State>
		static constexpr void for_each(TopStruct top_struct, F f, State state) noexcept {
			std::size_t len = top_struct.template length<Dim>(state);
			for(std::size_t i = 0; i < len; i++)
				for_each_impl<RetSig>::for_each(top_struct, f, state.template with<index_in<Dim>>(i));
		}
	};
	template<char Dim, class... RetSigs>
	struct for_each_impl<dep_function_sig<Dim, RetSigs...>> {
		template<class TopStruct, class F, class State>
		static constexpr void for_each(TopStruct top_struct, F f, State state) noexcept {
			for_each(top_struct, f, state, std::index_sequence_for<RetSigs...>());
		}
		template<class TopStruct, class F, class State, std::size_t... I>
		static constexpr void for_each(TopStruct top_struct, F f, State state, std::index_sequence<I...>) noexcept {
			((void) 0, ..., for_each_impl<RetSigs>::for_each(top_struct, f, state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>())));
		}
	};
	template<class ValueType>
	struct for_each_impl<scalar_sig<ValueType>> {
		template<class TopStruct, class F, class State>
		static constexpr void for_each(TopStruct top_struct, F f, State state) noexcept {
			Struct::single_iter(f, state_at<Struct>(top_struct, state));
		}
	};
};

template<class... Structs>
constexpr traverser_t<union_t<Structs...>, unit_struct_t> traverser(Structs... s) noexcept { return traverser_t<union_t<Structs...>, unit_struct_t>(union_t<Structs...>(s...), unit_struct); }

} // namespace noarr

#endif // NOARR_STRUCTURES_TRAVERSER_HPP
