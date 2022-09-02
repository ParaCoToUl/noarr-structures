#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "blocks.hpp"
#include "reorder.hpp"

namespace noarr {

template<char Dim, char DimMajor, char DimMinor, class... OptionalMinorLengthT>
constexpr auto strip_mine(OptionalMinorLengthT... optional_minor_length) {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
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

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP
