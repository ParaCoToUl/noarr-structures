#ifndef NOARR_STRUCTURES_STATE_HPP
#define NOARR_STRUCTURES_STATE_HPP

#include <concepts>
#include <cstddef>
#include <type_traits>

#include "contain.hpp"
#include "utility.hpp"

namespace noarr {

template<typename T>
concept IsTag = requires(T a) {
	requires IsDimSequence<typename T::dims>;
	{ T::template all_accept<dim_accepter> } -> std::convertible_to<bool>;
	{ T::template any_accept<dim_accepter> } -> std::convertible_to<bool>;
	typename T::template map<dim_identity_mapper>;
};

template<class... Ts>
concept IsTagPack = (... && IsTag<Ts>);

template<auto Dim>
requires IsDim<decltype(Dim)>
struct length_in {
	using dims = dim_sequence<Dim>;

	template<class Pred>
	static constexpr bool all_accept = Pred::template value<Dim>;

	template<class Pred>
	static constexpr bool any_accept = Pred::template value<Dim>;

	template<class Fn>
	using map = length_in<Fn::template value<Dim>>;
};

template<auto Dim>
requires IsDim<decltype(Dim)>
struct index_in {
	using dims = dim_sequence<Dim>;

	template<class Pred>
	static constexpr bool all_accept = Pred::template value<Dim>;

	template<class Pred>
	static constexpr bool any_accept = Pred::template value<Dim>;

	template<class Fn>
	using map = index_in<Fn::template value<Dim>>;
};

template<IsTag Tag, class ValueType>
struct state_item {
	using tag = Tag;
	using value_type = ValueType;
};

template<class T>
struct is_state_item : std::false_type {};

template<IsTag Tag, class ValueType>
struct is_state_item<state_item<Tag, ValueType>> : std::true_type {};

template<class T>
static constexpr bool is_state_item_v = is_state_item<T>::value;

template<class T>
concept IsStateItem = is_state_item_v<std::remove_cvref_t<T>>;

namespace helpers {

template<class... StateItems>
requires (... && IsStateItem<StateItems>)
struct state_items_pack {
	template<IsStateItem HeadStateItem>
	using prepend = state_items_pack<HeadStateItem, StateItems...>;
};

template<IsTag Tag, class... StateItems>
struct state_index_of;

template<IsTag Tag, class ValueType, class... TailStateItems>
struct state_index_of<Tag, state_item<Tag, ValueType>, TailStateItems...> {
	static constexpr auto result = some<std::size_t>{0};
};

template<IsTag Tag, class HeadStateItem, class... TailStateItems>
struct state_index_of<Tag, HeadStateItem, TailStateItems...> {
	static constexpr auto result =
		state_index_of<Tag, TailStateItems...>::result.and_then([](auto v) constexpr noexcept { return v + 1; });
};

template<IsTag Tag>
struct state_index_of<Tag> {
	static constexpr auto result = none{};
};

template<IsTag Tag, class... StateItems>
struct state_remove_item;

template<IsTag Tag, class ValueType, class... TailStateItems>
struct state_remove_item<Tag, state_item<Tag, ValueType>, TailStateItems...> {
	using result = typename state_remove_item<Tag, TailStateItems...>::result;
};

template<IsTag Tag, class HeadStateItem, class... TailStateItems>
struct state_remove_item<Tag, HeadStateItem, TailStateItems...> {
	using result = typename state_remove_item<Tag, TailStateItems...>::result::template prepend<HeadStateItem>;
};

template<IsTag Tag>
struct state_remove_item<Tag> {
	using result = state_items_pack<>;
};

template<class StateItemsPack, class... Tags>
struct state_remove_items;

template<class StateItemsPack>
struct state_remove_items<StateItemsPack> {
	using result = StateItemsPack;
};

template<class... StateItems, class Tag>
struct state_remove_items<state_items_pack<StateItems...>, Tag> {
	using result = typename state_remove_item<Tag, StateItems...>::result;
};

template<class... StateItems, class Tag, class... Tags>
struct state_remove_items<state_items_pack<StateItems...>, Tag, Tags...> {
	using recursion_result = typename state_remove_item<Tag, StateItems...>::result;
	using result = typename state_remove_items<recursion_result, Tags...>::result;
};

template<class Pred, class... StateItems>
struct state_filter_item;

template<class Pred, IsTag Tag, class ValueType, class... TailStateItems>
requires (Pred::template value<Tag>)
struct state_filter_item<Pred, state_item<Tag, ValueType>, TailStateItems...> {
	using result =
		typename state_filter_item<Pred, TailStateItems...>::result::template prepend<state_item<Tag, ValueType>>;
};

template<class Pred, class HeadStateItem, class... TailStateItems>
struct state_filter_item<Pred, HeadStateItem, TailStateItems...> {
	using result = typename state_filter_item<Pred, TailStateItems...>::result;
};

template<class Pred>
struct state_filter_item<Pred> {
	using result = state_items_pack<>;
};

template<class StateItemsPack, class Pred>
struct state_filter_items;

template<class... StateItems, class Pred>
struct state_filter_items<state_items_pack<StateItems...>, Pred> {
	using result = typename state_filter_item<Pred, StateItems...>::result;
};

} // namespace helpers

template<class... StateItems>
struct state : strict_contain<typename StateItems::value_type...> {
	using base = strict_contain<typename StateItems::value_type...>;
	using base::base;

	template<class Tag>
	requires IsTag<Tag>
	static constexpr auto index_of = helpers::state_index_of<Tag, StateItems...>::result;

	template<class... Tags>
	requires IsTagPack<Tags...>
	static constexpr bool contains = (... && index_of<Tags>.present);

	static constexpr bool is_empty = sizeof...(StateItems) == 0;

	using items_pack = helpers::state_items_pack<StateItems...>;

	template<class Tag>
	requires IsTag<Tag> && contains<Tag>
	[[nodiscard]]
	constexpr auto get() const noexcept {
		return base::template get<index_of<Tag>.value>();
	}

	template<class... KeptStateItems>
	[[nodiscard]]
	constexpr state<KeptStateItems...>
	items_restrict(helpers::state_items_pack<KeptStateItems...> /*kept*/ = {}) const noexcept {
		return state<KeptStateItems...>(get<typename KeptStateItems::tag>()...);
	}

	template<class... NewTags, class... NewValueTypes, class... KeptStateItems>
	requires IsTagPack<NewTags...>
	[[nodiscard]]
	constexpr state<KeptStateItems..., state_item<NewTags, NewValueTypes>...>
	items_restrict_add(helpers::state_items_pack<KeptStateItems...> /*kept*/,
	                   NewValueTypes... new_values) const noexcept {
		return state<KeptStateItems..., state_item<NewTags, NewValueTypes>...>(get<typename KeptStateItems::tag>()...,
		                                                                       new_values...);
	}

	template<class... Tags>
	requires IsTagPack<Tags...>
	[[nodiscard]]
	constexpr auto remove() const noexcept {
		return items_restrict(typename helpers::state_remove_items<items_pack, Tags...>::result());
	}

private:
	template<class... Tags>
	struct filter_predicate {
		template<class Tag>
		static constexpr bool value = (... || std::is_same_v<Tag, Tags>);
	};

public:
	template<class... Tags>
	requires IsTagPack<Tags...>
	[[nodiscard]]
	constexpr auto filter() const noexcept {
		return items_restrict(typename helpers::state_filter_items<items_pack, filter_predicate<Tags...>>::result());
	}

	template<class... Tags, class... ValueTypes>
	requires IsTagPack<Tags...>
	[[nodiscard]]
	constexpr auto with(ValueTypes... values) const noexcept {
		return items_restrict_add<Tags...>(typename helpers::state_remove_items<items_pack, Tags...>::result(),
		                                   values...);
	}
};

template<class T>
struct is_state_impl : std::false_type {};

template<class... T>
struct is_state_impl<state<T...>> : std::true_type {};

template<class T>
using is_state = is_state_impl<std::remove_cvref_t<T>>;

template<class T>
constexpr bool is_state_v = is_state<T>::value;

template<class T>
concept IsState = is_state_v<std::remove_cvref_t<T>>;

template<class State, auto Dim>
concept HasNotSetIndex = IsState<State> && !State::template contains<index_in<Dim>>;

template<class State, auto Dim>
concept HasSetIndex = IsState<State> && State::template contains<index_in<Dim>>;

template<IsState State, IsTag Tag>
using state_get_t = decltype(std::declval<State>().template get<Tag>());

template<IsState State, class... Tags>
requires IsTagPack<Tags...>
using state_remove_t = decltype(std::declval<State>().template remove<Tags...>());

static constexpr state<> empty_state;

template<class T>
struct to_state;

template<IsState T>
struct to_state<T> {
	using type = std::remove_cvref_t<T>;

	[[nodiscard]]
	static constexpr type convert(T t) noexcept {
		return t;
	}
};

template<class T>
constexpr auto convert_to_state(T &&t) noexcept {
	return to_state<std::remove_cvref_t<T>>::convert(std::forward<T>(t));
}

template<class T>
concept ToState = requires(T t) {
	{ convert_to_state(t) } -> IsState;
};

namespace helpers {

constexpr std::size_t supported_index_type(std::size_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_index_type(std::integral_constant<std::size_t, Value>);

constexpr std::size_t supported_diff_type(std::size_t);
constexpr std::ptrdiff_t supported_diff_type(std::ptrdiff_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_diff_type(std::integral_constant<std::size_t, Value>);
template<std::ptrdiff_t Value>
constexpr std::integral_constant<std::ptrdiff_t, Value>
	supported_diff_type(std::integral_constant<std::ptrdiff_t, Value>);

constexpr std::size_t supported_diff_index_type(std::size_t);
constexpr std::size_t supported_diff_index_type(std::ptrdiff_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value>
	supported_diff_index_type(std::integral_constant<std::size_t, Value>);
template<std::ptrdiff_t Value>
constexpr std::integral_constant<std::size_t, static_cast<std::size_t>(Value)>
	supported_diff_index_type(std::integral_constant<std::ptrdiff_t, Value>);

} // namespace helpers

template<class T>
using good_index_t = decltype(helpers::supported_index_type(std::declval<T>()));

template<class T>
using good_diff_t = decltype(helpers::supported_diff_type(std::declval<T>()));

template<class T>
using good_diff_index_t = decltype(helpers::supported_diff_index_type(std::declval<T>()));

template<class... Tags, class... ValueTypes>
requires IsTagPack<Tags...>
constexpr auto make_state(ValueTypes... values) noexcept {
	return state<state_item<Tags, good_index_t<ValueTypes>>...>(values...);
}

template<class... StateItemsA, class... StateItemsB>
constexpr auto operator&(state<StateItemsA...> state_a, [[maybe_unused]] state<StateItemsB...> state_b) noexcept {
	return state_a.template with<typename StateItemsB::tag...>(state_b.template get<typename StateItemsB::tag>()...);
}

template<class... StateItems>
constexpr state<StateItems...> operator+(state<StateItems...> state_a) noexcept {
	return state_a;
}

template<class... StateItems>
constexpr state<StateItems...> operator-([[maybe_unused]] state<StateItems...> state_a) noexcept {
	using namespace noarr::constexpr_arithmetic;
	return state<StateItems...>(-state_a.template get<typename StateItems::tag>()...);
}

template<class... StateItems>
constexpr state<StateItems...> operator+(state<StateItems...> state_a, state<StateItems...> state_b) noexcept {
	using namespace noarr::constexpr_arithmetic;
	return state<StateItems...>(
		(state_a.template get<typename StateItems::tag>() + state_b.template get<typename StateItems::tag>())...);
}

template<class... StateItems>
constexpr state<StateItems...> operator+(state<StateItems...> state_a, state<> /*state_b*/) noexcept {
	return state_a;
}

template<class... StateItems>
constexpr state<StateItems...> operator+(state<> /*state_a*/, state<StateItems...> state_b) noexcept {
	return state_b;
}

constexpr state<> operator+(state<> /*state_a*/, state<> /*state_b*/) noexcept { return empty_state; }

template<class... StateItemsA, class... StateItemsB>
constexpr auto operator+(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	// items that are in just one of the states
	const auto base =
		state_a.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
	                                                                typename StateItemsB::tag...>::result()) &
		state_b.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsB...>,
	                                                                typename StateItemsA::tag...>::result());

	return [=]<class... StateItems>(state<StateItems...> base) constexpr noexcept {
		// items that are in both states
		const auto added =
			state_a.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
		                                                                typename StateItems::tag...>::result()) +
			state_b.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
		                                                                typename StateItems::tag...>::result());

		return added & base;
	}(base);
}

template<class... StateItems>
constexpr state<StateItems...> operator-(state<StateItems...> state_a, state<StateItems...> state_b) noexcept {
	using namespace noarr::constexpr_arithmetic;
	return state<StateItems...>(state_a.template get<typename StateItems::tag>() -
	                            state_b.template get<typename StateItems::tag>()...);
}

template<class... StateItems>
constexpr state<StateItems...> operator-(state<StateItems...> state_a, state<> /*state_b*/) noexcept {
	return state_a;
}

template<class... StateItems>
constexpr state<StateItems...> operator-(state<> /*state_a*/, state<StateItems...> state_b) noexcept {
	return -state_b;
}

constexpr state<> operator-(state<> /*state_a*/, state<> /*state_b*/) noexcept { return empty_state; }

template<class... StateItemsA, class... StateItemsB>
constexpr auto operator-(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	const auto base =
		state_a.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
	                                                                typename StateItemsB::tag...>::result()) &
		-state_b.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsB...>,
	                                                                 typename StateItemsA::tag...>::result());

	return [=]<class... StateItems>(state<StateItems...> base) constexpr noexcept {
		const auto added =
			state_a.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
		                                                                typename StateItems::tag...>::result()) -
			state_b.items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItemsA...>,
		                                                                typename StateItems::tag...>::result());

		return added & base;
	}(base);
}

template<class... StateItemsA, class... StateItemsB>
constexpr bool operator==([[maybe_unused]] state<StateItemsA...> state_a,
                          [[maybe_unused]] state<StateItemsB...> state_b) noexcept {
	if constexpr (state<StateItemsB...>::template contains<typename StateItemsA::tag...> &&
	              state<StateItemsA...>::template contains<typename StateItemsB::tag...>) {
		return (... && (state_a.template get<typename StateItemsA::tag>() ==
		                state_b.template get<typename StateItemsA::tag>()));
	} else {
		return false;
	}
}

template<class... StateItemsA, class... StateItemsB>
constexpr bool operator<=([[maybe_unused]] state<StateItemsA...> state_a,
                          [[maybe_unused]] state<StateItemsB...> state_b) noexcept {
	if constexpr ((... && state<StateItemsB...>::template contains<typename StateItemsA::tag>)) {
		return (... && (state_a.template get<typename StateItemsA::tag>() <=
		                state_b.template get<typename StateItemsA::tag>()));
	} else {
		return false;
	}
}

template<class... StateItemsA, class... StateItemsB>
constexpr bool operator>=(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	return state_b <= state_a;
}

template<class... StateItemsA, class... StateItemsB>
constexpr bool operator<(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	return state_a <= state_b && state_a != state_b;
}

template<class... StateItemsA, class... StateItemsB>
constexpr bool operator>(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	return state_b < state_a;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STATE_HPP
