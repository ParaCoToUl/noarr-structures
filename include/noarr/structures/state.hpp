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

// TODO specialize to fix and set_length
template<class StateUpdates, class Struct>
struct setter_t : contain<StateUpdates, Struct> {
	using base = contain<StateUpdates, Struct>;
	using base::base;

	constexpr StateUpdates state_update() const noexcept { return base::template get<0>(); }
	constexpr Struct sub_structure() const noexcept { return base::template get<1>(); }
	constexpr std::tuple<Struct> sub_structures() { return std::tuple(sub_structure()); }

private:
	template<char Dim, class ValueType>
	struct replacers {
		template<class Original>
		struct index_dim_replacement;
		template<class ArgLength, class RetType>
		struct index_dim_replacement<function_type<Dim, ArgLength, RetType>> {
			static_assert(ArgLength::is_known, "Set length before setting the value");
			using type = RetType;
		};
		template<class... RetTypes>
		struct index_dim_replacement<dep_function_type<Dim, RetTypes...>> {
			using original = dep_function_type<Dim, RetTypes...>;
			static_assert(ValueType::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
			using type = typename dep_function_type<Dim, RetTypes...>::ret_type<ValueType::value>;
		};
		template<class Original>
		struct length_dim_replacement {
			static_assert(!Original::dependent, "Cannot set tuple length");
			static_assert(!Original::arg_length::is_known, "Length already set");
			using type = function_type<Dim, arg_length_from_t<ValueType>, typename Original::ret_type>;
		};
	};
	template<class StateUpdatesTail = StateUpdates, class = void>
	struct helper;
	template<char Dim, class ValueType, class... StateItems>
	struct helper<state<state_item<index_in<Dim>, ValueType>, StateItems...>> {
		using result = typename helper<state<StateItems...>>::result::replace<replacers<Dim, ValueType>::template index_dim_replacement, Dim>;
	};
	template<char Dim, class ValueType, class... StateItems>
	struct helper<state<state_item<length_in<Dim>, ValueType>, StateItems...>> {
		using result = typename helper<state<StateItems...>>::result::replace<replacers<Dim, ValueType>::template length_dim_replacement, Dim>;
	};
	template<class Useless>
	struct helper<state<>, Useless> {
		using result = typename Struct::struct_type;
	};
public:
	using struct_type = typename helper<>::result;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		// TODO check absence of new items
		return sub_structure().size(state.merge(state_update()));
	}
};

template<class StateUpdates>
struct setter : contain<StateUpdates> {
	explicit constexpr setter(StateUpdates u) noexcept : contain<StateUpdates>(u) {}

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return setter_t<StateUpdates, Struct>(contain<StateUpdates>::template get<0>(), s); }
};

static constexpr state<> empty_state;

} // namespace noarr

#endif // NOARR_STRUCTURES_STATE_HPP
