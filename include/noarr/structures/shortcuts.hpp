#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "layouts.hpp"
#include "blocks.hpp"
#include "reorder.hpp"
#include "setters.hpp"

namespace noarr {

// Common compositions

template<char Dim>
constexpr auto sized_vector(std::size_t length) {
	return vector<Dim>() ^ set_length<Dim>(length);
}

template<char Dim, char DimMajor, char DimMinor, class MinorLengthT>
constexpr auto into_blocks(MinorLengthT minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>() ^ set_length<DimMinor>(minor_length);
}

template<char DimMajor, char DimMinor, char Dim, class MinorSizeT>
constexpr auto merge_blocks(MinorSizeT minor_length) {
	return set_length<DimMinor>(minor_length) ^ merge_blocks<DimMajor, DimMinor, Dim>();
}

template<char Dim, char DimMajor, char DimMinor, class... OptionalMinorLengthT>
constexpr auto strip_mine(OptionalMinorLengthT... optional_minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
}

template<char Dim, char TmpDim = 127, class StartT, class StrideT>
constexpr auto step(StartT start, StrideT stride) {
	return into_blocks<Dim, Dim, TmpDim>(stride) ^ fix<TmpDim>(start);
}

// Working with state (especially in traverser lambdas)

template<char Dim, class State>
constexpr auto get_index(State state) {
	return state.template get<index_in<Dim>>();
}

template<char Dim, class State, class F>
constexpr auto update_index(State state, F f) {
	static_assert(State::template contains<index_in<Dim>>, "Requested dimension does not exist. To add a new dimension instead of updating existing one, use .template with<index_in<'...'>>(...)");
	auto new_index = f(state.template get<index_in<Dim>>());
	return state.template remove<index_in<Dim>>().template with<index_in<Dim>>(good_index_t<decltype(new_index)>(new_index));
}

namespace helpers {

using ssize_t = std::make_signed_t<std::size_t>;

template<char>
using always_ssize_t = ssize_t;

} // namespace helpers

template<char... Dims, class State>
constexpr auto neighbor(State state, helpers::always_ssize_t<Dims>... diffs) noexcept {
	static_assert((... && State::template contains<index_in<Dims>>), "Requested dimension does not exist");
	static_assert((... && std::is_same_v<state_get_t<State, index_in<Dims>>, std::size_t>), "Cannot shift in a dimension that is not dynamic");
	return state.template remove<index_in<Dims>...>().template with<index_in<Dims>...>(std::size_t((helpers::ssize_t) state.template get<index_in<Dims>>() + diffs)...);
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
