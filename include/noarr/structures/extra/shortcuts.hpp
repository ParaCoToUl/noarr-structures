#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "../base/utility.hpp"
#include "../structs/bcast.hpp"
#include "../structs/layouts.hpp"
#include "../structs/blocks.hpp"
#include "../structs/views.hpp"
#include "../structs/setters.hpp"

namespace noarr {

// Common compositions

template<char Dim, class LenT>
constexpr auto sized_vector(LenT length) {
	return vector<Dim>() ^ set_length<Dim>(length);
}

template<char Dim, class Struct, class State>
constexpr auto length_like(Struct structure, State state) {
	return set_length<Dim>(structure | get_length<Dim>(state));
}

template<char Dim, class Struct>
constexpr auto length_like(Struct structure) {
	return length_like<Dim>(structure, empty_state);
}

template<char ...Dims, class Struct, class State>
constexpr auto lengths_like(Struct structure, State state) {
	return (... ^ length_like<Dims>(structure, state));
}

template<char ...Dims, class Struct>
constexpr auto lengths_like(Struct structure) {
	return lengths_like<Dims...>(structure, empty_state);
}

template<char Dim, class Struct, class State>
constexpr auto vector_like(Struct structure, State state) {
	return (vector<Dim>() ^ length_like<Dim>(structure, state));
}

template<char Dim, class Struct>
constexpr auto vector_like(Struct structure) {
	return vector_like<Dim>(structure, empty_state);
}

template<char ...Dims, class Struct, class State>
constexpr auto vectors_like(Struct structure, State state) {
	return (... ^ (vector_like<Dims>(structure, state)));
}

template<char ...Dims, class Struct>
constexpr auto vectors_like(Struct structure) {
	return vectors_like<Dims...>(structure, empty_state);
}

template<char Dim, char DimMajor, char DimMinor, class MinorLengthT>
constexpr auto into_blocks(MinorLengthT minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>() ^ set_length<DimMinor>(minor_length);
}

template<char Dim, char DimMajor, char DimMinor, char DimIsPresent, class MinorLengthT>
constexpr auto into_blocks_dynamic(MinorLengthT minor_length) {
	return into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>() ^ set_length<DimMinor>(minor_length);
}

template<char DimMajor, char DimMinor, char Dim, class MinorSizeT>
constexpr auto merge_blocks(MinorSizeT minor_length) {
	return set_length<DimMinor>(minor_length) ^ merge_blocks<DimMajor, DimMinor, Dim>();
}

template<char Dim, char DimMajor, char DimMinor, class... OptionalMinorLengthT>
constexpr auto strip_mine(OptionalMinorLengthT... optional_minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
}

template<char ...Dims, class ...Sizes>
constexpr auto bcast(Sizes ...lengths) {
	return (... ^ (bcast<Dims>() ^ set_length<Dims>(lengths)));
}

// Working with state (especially in traverser lambdas)

template<char Dim, class State>
constexpr auto get_index(State state) {
	return state.template get<index_in<Dim>>();
}

template<char... Dim, class State>
constexpr auto get_indices(State state) {
	return std::make_tuple(state.template get<index_in<Dim>>()...);
}

template<char... Dim, class... ValueType>
constexpr auto idx(ValueType... value) {
	return state<state_item<index_in<Dim>, good_index_t<ValueType>>...>(value...);
}

template<char Dim, class State, class F>
constexpr auto update_index(State state, F f) {
	static_assert(State::template contains<index_in<Dim>>, "Requested dimension does not exist. To add a new dimension instead of updating existing one, use .template with<index_in<'...'>>(...)");
	auto new_index = f(state.template get<index_in<Dim>>());
	return state.template with<index_in<Dim>>(good_index_t<decltype(new_index)>(new_index));
}

template<class State>
constexpr auto getter(State state) noexcept { return [state](auto ptr) constexpr noexcept {
	return get_at<State, decltype(ptr)>(ptr, state);
}; }

template<char... Dims, class State>
constexpr auto neighbor(State state, std::enable_if_t<true || Dims, std::ptrdiff_t>... diffs) noexcept {
	static_assert((... && State::template contains<index_in<Dims>>), "Requested dimension does not exist");
	static_assert((... && std::is_same_v<state_get_t<State, index_in<Dims>>, std::size_t>), "Cannot shift in a dimension that is not dynamic");
	return state.template with<index_in<Dims>...>(std::size_t(state.template get<index_in<Dims>>() + diffs)...);
}

// State to structure

template<char... Dim, class... IdxT>
constexpr auto fix(state<state_item<index_in<Dim>, IdxT>...> state) noexcept {
	return fix<Dim...>(get_index<Dim>(state)...);
}
template<class... StateItem>
constexpr void fix(state<StateItem...>) {
	static_assert(always_false<state<StateItem...>>, "Unrecognized items in state. The fix(state) shortcut can be only used to fix indices (index_in<...>)");
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP
