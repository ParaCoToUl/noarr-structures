#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include <cstddef>

#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/funcs.hpp"
#include "../extra/sig_utils.hpp"
#include "../extra/to_struct.hpp"
#include "../extra/traverser.hpp"
#include "../structs/bcast.hpp"
#include "../structs/blocks.hpp"
#include "../structs/layouts.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"
#include "../structs/views.hpp"

namespace noarr {

// Common compositions

template<auto... Dims>
requires IsDimPack<decltype(Dims)...>
constexpr auto vectors() noexcept {
	return (... ^ vector<Dims>());
}

template<IsDim auto Dim, class Length>
constexpr auto vector(Length length) noexcept {
	return vector<Dim>() ^ set_length<Dim>(length);
}

template<IsDim auto Dim, class Length>
[[deprecated("Use vector instead")]]
constexpr auto sized_vector(Length length) noexcept {
	return vector<Dim>() ^ set_length<Dim>(length);
}

template<auto... Dims, class... Lengths>
requires IsDimPack<decltype(Dims)...> && (sizeof...(Dims) == sizeof...(Lengths))
constexpr auto vectors(Lengths... lengths) noexcept {
	return (... ^ vector<Dims>(lengths));
}

template<auto... Dims, class... Lengths>
requires IsDimPack<decltype(Dims)...> && (sizeof...(Dims) == sizeof...(Lengths))
[[deprecated("Use vectors instead")]] constexpr auto sized_vectors(Lengths... lengths) noexcept {
	return (... ^ vector<Dims>(lengths));
}

template<IsDim auto Dim, std::size_t L>
struct array_proto {
	static constexpr bool proto_preserves_layout = false;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return s ^ vector<Dim>() ^ set_length<Dim>(lit<L>);
	}
};

template<IsDim auto Dim, std::size_t L, class SubStruct>
using array_t = decltype(std::declval<SubStruct>() ^ array_proto<Dim, L>());

template<IsDim auto Dim, std::size_t L>
constexpr auto array() noexcept {
	return array_proto<Dim, L>();
}

template<IsDim auto Dim, IsStruct Struct, IsState State>
constexpr auto length_like(Struct structure, State state) noexcept {
	return set_length<Dim>(structure | get_length<Dim>(state));
}

template<IsDim auto Dim, IsStruct Struct>
constexpr auto length_like(Struct structure) noexcept {
	return length_like<Dim>(structure, empty_state);
}

template<auto... Dims, class Struct, class State>
requires IsDimPack<decltype(Dims)...> && IsStruct<Struct> && IsState<State>
constexpr auto lengths_like(Struct structure, State state) noexcept {
	return (... ^ length_like<Dims>(structure, state));
}

template<auto... Dims, class Struct>
requires IsDimPack<decltype(Dims)...> && IsStruct<Struct>
constexpr auto lengths_like(Struct structure) noexcept {
	return lengths_like<Dims...>(structure, empty_state);
}

template<auto Dim, auto... Dims, class Traverser>
requires IsDimPack<decltype(Dim), decltype(Dims)...> && ToTraverser<Traverser>
constexpr auto lengths_like(const Traverser &traverser) noexcept {
	return lengths_like<Dim, Dims...>(convert_to_traverser(traverser).top_struct());
}

template<IsDim auto Dim, IsStruct Struct, IsState State>
constexpr auto vector_like(Struct structure, State state) noexcept {
	return vector<Dim>() ^ length_like<Dim>(structure, state);
}

template<IsDim auto Dim, IsStruct Struct>
constexpr auto vector_like(Struct structure) noexcept {
	return vector_like<Dim>(structure, empty_state);
}

template<auto Dim, auto... Dims, class Struct, IsState State>
requires IsDimPack<decltype(Dim), decltype(Dims)...> && IsStruct<Struct> && IsState<State>
constexpr auto vectors_like(Struct structure, State state) noexcept {
	return (vector_like<Dim>(structure, state) ^ ... ^ vector_like<Dims>(structure, state));
}

template<auto Dim, auto... Dims, class Struct>
requires IsDimPack<decltype(Dim), decltype(Dims)...> && IsStruct<Struct>
constexpr auto vectors_like(Struct structure) noexcept {
	return vectors_like<Dim, Dims...>(structure, empty_state);
}

template<class Struct, class State>
requires IsState<State>
constexpr auto vectors_like(Struct structure, State state) noexcept {
	using signature = typename to_struct<Struct>::type::signature;
	using dim_seq = sig_dim_seq<signature>;

	return [structure, state]<auto... Dims>(dim_sequence<Dims...>) noexcept {
		return vectors_like<Dims...>(structure, state);
	}(dim_seq());
}

template<IsStruct Struct>
constexpr auto vectors_like(Struct structure) noexcept {
	return vectors_like<Struct>(structure, empty_state);
}

template<ToTraverser Traverser>
constexpr auto vectors_like(const Traverser &traverser) noexcept {
	return vectors_like(convert_to_traverser(traverser).top_struct());
}

template<auto Dim, auto... Dims, class Traverser>
requires IsDimPack<decltype(Dim), decltype(Dims)...> && ToTraverser<Traverser>
constexpr auto vectors_like(const Traverser &traverser) noexcept {
	return vectors_like<Dim, Dims...>(convert_to_traverser(traverser).top_struct());
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor = Dim>
constexpr auto into_blocks(auto minor_length) noexcept {
	return into_blocks<Dim, DimMajor, DimMinor>() ^ set_length<DimMinor>(minor_length);
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
constexpr auto into_blocks_dynamic(auto minor_length) noexcept {
	return into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>() ^ set_length<DimMinor>(minor_length);
}

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim = DimMinor>
constexpr auto merge_blocks(auto minor_length) noexcept {
	return set_length<DimMinor>(minor_length) ^ merge_blocks<DimMajor, DimMinor, Dim>();
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor = Dim>
constexpr auto strip_mine(auto... optional_minor_length) noexcept {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
constexpr auto strip_mine_dynamic(auto... optional_minor_length) noexcept {
	return into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>(optional_minor_length...) ^ hoist<DimMajor>();
}

template<auto... Dims, class... Lengths>
constexpr auto bcast(Lengths... lengths) noexcept
requires (sizeof...(Dims) == sizeof...(Lengths)) && IsDimPack<decltype(Dims)...>
{
	return (... ^ (bcast<Dims>() ^ set_length<Dims>(lengths)));
}

// Working with state (especially in traverser lambdas)

template<IsDim auto Dim, ToState HasState>
constexpr auto get_index(HasState has_state) noexcept {
	return convert_to_state(has_state).template get<index_in<Dim>>();
}

template<IsDim auto Dim, ToState HasState>
constexpr auto get_length_in(HasState has_state) noexcept {
	return convert_to_state(has_state).template get<length_in<Dim>>();
}

template<auto... Dims, class HasState>
requires IsDimPack<decltype(Dims)...> && ToState<HasState>
constexpr auto get_indices(HasState has_state) noexcept {
	return std::make_tuple(convert_to_state(has_state).template get<index_in<Dims>>()...);
}

template<auto... Dims, class HasState>
requires IsDimPack<decltype(Dims)...> && ToState<HasState>
constexpr auto get_lengths_in(HasState has_state) noexcept {
	return std::make_tuple(convert_to_state(has_state).template get<length_in<Dims>>()...);
}

template<auto... Dims, class HasState>
requires IsDimPack<decltype(Dims)...> && ToState<HasState>
constexpr auto filter_indices(HasState has_state) noexcept {
	return convert_to_state(has_state).template filter<index_in<Dims>...>();
}

template<auto... Dims, class HasState>
requires IsDimPack<decltype(Dims)...> && ToState<HasState>
constexpr auto filter_lengths(HasState has_state) noexcept {
	return convert_to_state(has_state).template filter<length_in<Dims>...>();
}

template<auto... Dims, class... ValueType>
requires IsDimPack<decltype(Dims)...>
constexpr auto idx(ValueType... value) noexcept {
	return state<state_item<index_in<Dims>, good_index_t<ValueType>>...>(value...);
}

template<IsDim auto Dim, class F, IsState State>
constexpr auto update_index(State state, F f) noexcept {
	static_assert(state_contains<State, index_in<Dim>>,
	              "Requested dimension does not exist. To add a new dimension instead of updating existing one, use "
	              ".template with<index_in<'...'>>(...)");
	const auto new_index = f(state.template get<index_in<Dim>>());
	return state.template with<index_in<Dim>>(good_index_t<decltype(new_index)>(new_index));
}

template<auto... Dims, IsState State, class... Diffs>
requires (sizeof...(Dims) == sizeof...(Diffs)) && IsDimPack<decltype(Dims)...>
constexpr auto neighbor(State state, Diffs... diffs) noexcept {
	using namespace noarr::constexpr_arithmetic;
	static_assert((... && state_contains<State, index_in<Dims>>), "Requested dimension does not exist");
	static_assert((... && std::is_same_v<state_get_t<State, index_in<Dims>>, std::size_t>),
	              "Cannot shift in a dimension that is not dynamic");
	return state.template with<index_in<Dims>...>(
		good_diff_index_t<decltype(state.template get<index_in<Dims>>() + diffs)>(state.template get<index_in<Dims>>() +
	                                                                              diffs)...);
}

template<auto... Dims, class Struct, class... StateItems>
requires IsDimPack<decltype(Dims)...>
constexpr auto symmetric_span(Struct structure, state<StateItems...> state, auto offset) noexcept {
	return (... ^ span<Dims>(offset, (structure | get_length<Dims>(state)) - offset));
}

template<auto... Dims, class Struct>
requires IsDimPack<decltype(Dims)...>
constexpr auto symmetric_span(Struct structure, auto offset) noexcept {
	return symmetric_span<Dims...>(structure, empty_state, offset);
}

template<auto... Dims, class Struct, class... StateItems, class... Offsets>
requires (sizeof...(Dims) == sizeof...(Offsets)) && IsDimPack<decltype(Dims)...>
constexpr auto symmetric_spans(Struct structure, state<StateItems...> state, Offsets... offsets) noexcept {
	return (... ^ symmetric_span<Dims>(structure, state, offsets));
}

template<auto... Dims, class Struct, class... Offsets>
requires (sizeof...(Dims) == sizeof...(Offsets)) && IsDimPack<decltype(Dims)...>
constexpr auto symmetric_spans(Struct structure, Offsets... offsets) noexcept {
	return symmetric_spans<Dims...>(structure, empty_state, offsets...);
}

// State to structure

namespace helpers {

template<IsDim auto Dim, class IdxT, IsState State>
constexpr auto state_construct_fix(state_item<index_in<Dim>, IdxT> /*idx*/, State state) noexcept {
	return fix<Dim>(state);
}

template<class StateItem, IsState State>
constexpr auto state_construct_fix(StateItem /*idx*/, State /*state*/) noexcept {
	return neutral_proto();
}

template<IsDim auto Dim, class LenT, IsState State>
constexpr auto state_construct_set_length(state_item<length_in<Dim>, LenT> /*len*/, State state) noexcept {
	return set_length<Dim>(state);
}

template<class StateItem, IsState State>
constexpr auto state_construct_set_length(StateItem /*len*/, State /*state*/) noexcept {
	return neutral_proto();
}

template<IsStruct Struct>
struct index_space_size_impl {
public:
	template<class DimTree>
	static constexpr std::size_t get(Struct structure) noexcept {
		return get_impl(DimTree{}, structure);
	}

	static constexpr std::size_t get(Struct structure) noexcept {
		return get_impl(sig_dim_tree<typename Struct::signature>{}, structure);
	}

private:
	template<auto Dim, class... Branches>
	requires (sizeof...(Branches) != 1)
	static constexpr std::size_t get_impl(dim_tree<Dim, Branches...> /*dt*/, Struct structure) noexcept {
		return (... + get_impl(Branches{}, structure));
	}

	template<auto Dim, class Branch>
	static constexpr std::size_t get_impl(dim_tree<Dim, Branch> /*dt*/, Struct structure) noexcept {
		return (structure | get_length<Dim>()) * get_impl(Branch{}, structure);
	}

	static constexpr std::size_t get_impl(dim_sequence<> /*dt*/, Struct /*structure*/) noexcept { return 1; }
};

} // namespace helpers

template<class... StateItem>
constexpr auto fix([[maybe_unused]] state<StateItem...> state) noexcept {
	return (neutral_proto() ^ ... ^ helpers::state_construct_fix(StateItem(), state));
}

template<auto Dim, auto... Dims, class... StateItem>
requires IsDimPack<decltype(Dim), decltype(Dims)...>
constexpr auto fix(state<StateItem...> state) noexcept {
	return fix<Dim, Dims...>(get_index<Dim>(state), get_index<Dims>(state)...);
}

template<ToState HasState>
constexpr auto fix(HasState state) noexcept {
	return fix(convert_to_state(state));
}

template<auto Dim, auto... Dims, class HasState>
requires IsDimPack<decltype(Dim), decltype(Dims)...> && ToState<HasState>
constexpr auto fix(HasState state) noexcept {
	return fix<Dim, Dims...>(convert_to_state(state));
}

template<class... StateItem>
constexpr auto set_length([[maybe_unused]] state<StateItem...> state) noexcept {
	return (neutral_proto() ^ ... ^ helpers::state_construct_set_length(StateItem(), state));
}

template<ToState HasState>
constexpr auto set_length(HasState state) noexcept {
	return set_length(convert_to_state(state));
}

template<auto Dim, auto... Dims, class... StateItem>
requires IsDimPack<decltype(Dim), decltype(Dims)...>
constexpr auto set_length(state<StateItem...> state) noexcept {
	return (set_length<Dim>(get_length_in<Dim>(state)) ^ ... ^ set_length<Dims>(get_length_in<Dims>(state)));
}

template<class Struct>
constexpr std::size_t index_space_size(Struct structure) noexcept {
	return helpers::index_space_size_impl<Struct>::get(structure);
}

template<class DimTree, class Struct>
constexpr std::size_t index_space_size(Struct structure) noexcept {
	return helpers::index_space_size_impl<Struct>::template get<DimTree>(structure);
}

template<IsDim auto Dim, class Branch, class... Branches, IsState State = state<>>
constexpr auto fix_zeros(dim_tree<Dim, Branch, Branches...> /*unused*/, State state = empty_state) noexcept {
	return fix_zeros(Branch{}, state & idx<Dim>(lit<0>));
}

template<auto... Dims, IsState State = state<>>
requires IsDimPack<decltype(Dims)...>
constexpr auto fix_zeros(dim_sequence<Dims...> /*unused*/, State state = empty_state) noexcept {
	[[maybe_unused]]
	constexpr auto zero = [](auto /*Dim*/) { return lit<0>; };
	return state & idx<Dims...>(zero(Dims)...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP
