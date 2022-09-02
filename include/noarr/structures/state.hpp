#ifndef NOARR_STRUCTURES_STATE_HPP
#define NOARR_STRUCTURES_STATE_HPP

#include "contain.hpp"

namespace noarr {

template<char Dim>
struct length_in;

template<char Dim>
struct index_in;

template<class Tag, class ValueType>
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

	template<class Tag, class... StateItems>
	struct state_index_of;

	template<class Tag, class ValueType, class... TailStateItems>
	struct state_index_of<Tag, state_item<Tag, ValueType>, TailStateItems...> {
		static constexpr bool present = true;
		static constexpr std::size_t value = 0;
	};

	template<class Tag, class HeadStateItem, class... TailStateItems>
	struct state_index_of<Tag, HeadStateItem, TailStateItems...> {
		using recursion = state_index_of<Tag, TailStateItems...>;
		static constexpr bool present = recursion::present;
		static constexpr std::size_t value = 1 + recursion::value;
	};

	template<class Tag>
	struct state_index_of<Tag> {
		static constexpr bool present = false;
	};

	template<class Tag, class... StateItems>
	struct state_remove_item;

	template<class Tag, class ValueType, class... TailStateItems>
	struct state_remove_item<Tag, state_item<Tag, ValueType>, TailStateItems...> {
		using result = typename state_remove_item<Tag, TailStateItems...>::result;
	};

	template<class Tag, class HeadStateItem, class... TailStateItems>
	struct state_remove_item<Tag, HeadStateItem, TailStateItems...> {
		using result = typename state_remove_item<Tag, TailStateItems...>::result::prepend<HeadStateItem>;
	};

	template<class Tag>
	struct state_remove_item<Tag> {
		using result = state_items_pack<>;
	};

	template<class StateItemsPack, class... Tags>
	struct state_remove_items;

	template<class StateItemsPack>
	struct state_remove_items<StateItemsPack> {
		using result = StateItemsPack;
	};

	template<class... StateItems, class Tag, class... Tags>
	struct state_remove_items<state_items_pack<StateItems...>, Tag, Tags...> {
		using recursion_result = typename state_remove_item<Tag, StateItems...>::result;
		using result = typename state_remove_items<recursion_result, Tags...>::result;
	};
} // namespace helpers

template<class... StateItems>
struct state : contain<typename StateItems::value_type...> {
	using base = contain<typename StateItems::value_type...>;
	using base::base;

	template<class Tag>
	using index_of = helpers::state_index_of<Tag, StateItems...>;

	template<class Tag>
	static constexpr bool contains = index_of<Tag>::present;

	static constexpr bool is_empty = !sizeof...(StateItems);

	template<class Tag>
	constexpr decltype(auto) get() const noexcept {
		static_assert(contains<Tag>, "No such item");
		return base::template get<index_of<Tag>::value>();
	}

	template<class... NewStateItems>
	constexpr state<NewStateItems...> restrict(helpers::state_items_pack<NewStateItems...> = {}) const noexcept {
		return state<NewStateItems...>(get<typename NewStateItems::tag>()...);
	}

	template<class... Tags>
	constexpr auto remove() const noexcept {
		return restrict(typename helpers::state_remove_items<helpers::state_items_pack<StateItems...>, Tags...>::result());
	}

	template<class... Tags, class... ValueTypes>
	constexpr auto with(ValueTypes... values) const noexcept {
		return state<StateItems..., state_item<Tags, ValueTypes>...>(get<typename StateItems::tag>()..., values...);
	}

	template<class... NewStateItems>
	constexpr state<StateItems..., NewStateItems...> merge(const state<NewStateItems...> &other) const noexcept {
		return state<StateItems..., NewStateItems...>(get<typename StateItems::tag>()..., other.template get<typename NewStateItems::tag>()...);
	}

	template<class Tag>
	using get_t = decltype(std::declval<state>().template get<Tag>());

	template<class... Tags>
	using remove_t = decltype(std::declval<state>().template remove<Tags...>());
};

static constexpr state<> empty_state;

namespace helpers {

constexpr std::size_t supported_index_type(std::size_t);

template<std::size_t Value>
constexpr std::integral_constant<std::size_t, Value> supported_index_type(std::integral_constant<std::size_t, Value>);

} // namespace helpers

template<class T>
using good_index_t = decltype(helpers::supported_index_type(std::declval<T>()));

} // namespace noarr

#endif // NOARR_STRUCTURES_STATE_HPP
