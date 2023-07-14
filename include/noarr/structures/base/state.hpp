#ifndef NOARR_STRUCTURES_STATE_HPP
#define NOARR_STRUCTURES_STATE_HPP

#include "contain.hpp"
#include "utility.hpp"

namespace noarr {

template<typename T>
struct is_tag : std::false_type {};

template<typename T>
constexpr bool is_tag_v = is_tag<T>::value;

template<typename T>
concept IsTag = is_tag_v<T>;

template<IsDim auto Dim>
struct length_in;
template<IsDim auto Dim>
struct index_in;

template<IsDim auto Dim>
struct is_tag<length_in<Dim>> : std::true_type {};

template<IsDim auto Dim>
struct is_tag<index_in<Dim>> : std::true_type {};

template<IsTag Tag, class ValueType>
struct state_item {
	using tag = Tag;
	using value_type = ValueType;
};

namespace helpers {

template<class... StateItems>
struct state_items_pack {
	template<class HeadStateItem>
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
	static constexpr auto result = state_index_of<Tag, TailStateItems...>::result.and_then([](auto v){return v+1;});
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

template<class StateItemsPack, class... Tags> requires (... && IsTag<Tags>)
struct state_remove_items;

template<class StateItemsPack>
struct state_remove_items<StateItemsPack> {
	using result = StateItemsPack;
};

template<class... StateItems, class Tag, class... Tags> requires (IsTag<Tag> && ... && IsTag<Tags>)
struct state_remove_items<state_items_pack<StateItems...>, Tag, Tags...> {
	using recursion_result = typename state_remove_item<Tag, StateItems...>::result;
	using result = typename state_remove_items<recursion_result, Tags...>::result;
};

} // namespace helpers

template<class... StateItems>
struct state : contain<typename StateItems::value_type...> {
	using base = contain<typename StateItems::value_type...>;
	using base::base;

	template<class Tag> requires (IsTag<Tag>)
	static constexpr auto index_of = helpers::state_index_of<Tag, StateItems...>::result;

	template<class Tag> requires (IsTag<Tag>)
	static constexpr bool contains = index_of<Tag>.present;

	static constexpr bool is_empty = !sizeof...(StateItems);

	template<class Tag> requires (IsTag<Tag>)
	constexpr auto get() const noexcept {
		static_assert(contains<Tag>, "No such item");
		return base::template get<index_of<Tag>.value>();
	}

	template<class... KeptStateItems>
	constexpr state<KeptStateItems...> items_restrict(helpers::state_items_pack<KeptStateItems...> = {}) const noexcept {
		return state<KeptStateItems...>(get<typename KeptStateItems::tag>()...);
	}

	template<class... NewTags, class... NewValueTypes, class... KeptStateItems> requires (... && IsTag<NewTags>)
	constexpr state<KeptStateItems..., state_item<NewTags, NewValueTypes>...> items_restrict_add(helpers::state_items_pack<KeptStateItems...>, NewValueTypes... new_values) const noexcept {
		return state<KeptStateItems..., state_item<NewTags, NewValueTypes>...>(get<typename KeptStateItems::tag>()..., new_values...);
	}

	template<class... Tags> requires (... && IsTag<Tags>)
	constexpr auto remove() const noexcept {
		return items_restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItems...>, Tags...>::result());
	}

	template<class... Tags, class... ValueTypes> requires (... && IsTag<Tags>)
	constexpr auto with(ValueTypes... values) const noexcept {
		return items_restrict_add<Tags...>(typename helpers::state_remove_items<helpers::state_items_pack<StateItems...>, Tags...>::result(), values...);
	}
};

template<class T>
struct is_state : std::false_type {};

template<class... T>
struct is_state<state<T...>> : std::true_type {};

template<class T>
constexpr bool is_state_v = is_state<T>::value;

template<class T>
concept IsState = is_state_v<T>;

template<IsState State, IsTag Tag>
using state_get_t = decltype(std::declval<State>().template get<Tag>());

template<IsState State, class... Tags> requires (... && IsTag<Tags>)
using state_remove_t = decltype(std::declval<State>().template remove<Tags...>());

static constexpr state<> empty_state;

namespace helpers {

constexpr std::size_t supported_index_type(std::size_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_index_type(std::integral_constant<std::size_t, Value>);

constexpr std::size_t supported_diff_type(std::size_t);
constexpr std::ptrdiff_t supported_diff_type(std::ptrdiff_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_diff_type(std::integral_constant<std::size_t, Value>);
template<std::size_t Value>
constexpr std::integral_constant<std::ptrdiff_t, Value> supported_diff_type(std::integral_constant<std::ptrdiff_t, Value>);

constexpr std::size_t supported_diff_index_type(std::size_t);
constexpr std::size_t supported_diff_index_type(std::ptrdiff_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_diff_index_type(std::integral_constant<std::size_t, Value>);
template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_diff_index_type(std::integral_constant<std::ptrdiff_t, Value>);

} // namespace helpers

template<class T>
using good_index_t = decltype(helpers::supported_index_type(std::declval<T>()));

template<class T>
using good_diff_t = decltype(helpers::supported_diff_type(std::declval<T>()));

template<class T>
using good_diff_index_t = decltype(helpers::supported_diff_index_type(std::declval<T>()));



template<class... Tag, class... ValueType> requires (... && IsTag<Tag>)
constexpr auto make_state(ValueType... value) {
	return state<state_item<Tag, good_index_t<ValueType>>...>(value...);
}

template<class... StateItemsA, class... StateItemsB>
constexpr state<StateItemsA..., StateItemsB...> operator&(state<StateItemsA...> state_a, state<StateItemsB...> state_b) noexcept {
	return state<StateItemsA..., StateItemsB...>(state_a.template get<typename StateItemsA::tag>()..., state_b.template get<typename StateItemsB::tag>()...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_STATE_HPP
