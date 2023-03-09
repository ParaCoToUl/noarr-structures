#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "../base/utility.hpp"
#include "../structs/bcast.hpp"
#include "../structs/layouts.hpp"
#include "../structs/blocks.hpp"
#include "../structs/views.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"

namespace noarr {

// Common compositions

template<char Dim, class LenT>
constexpr auto sized_vector(LenT length) {
	return vector<Dim>() ^ set_length<Dim>(length);
}

template<char Dim, std::size_t L, class SubStruct = void>
struct array_impl {
	using type = decltype(std::declval<SubStruct>() ^ sized_vector<Dim>(lit<L>));
};

template<char Dim, std::size_t L>
struct array_impl<Dim, L, void> {
	using type = decltype(sized_vector<Dim>(lit<L>));
};

template<char Dim, std::size_t L, class SubStruct = void>
using array = typename array_impl<Dim, L, SubStruct>::type;

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

template<char ...Dims, class Struct, class Offset, class ...StateItems>
constexpr auto symmetric_slice(Struct structure, state<StateItems...> state, Offset offset) noexcept {
	using namespace constexpr_arithmetic;
	return (... ^ slice<Dims>(offset, (structure | get_length<Dims>(state)) - make_const<2>() * offset));
}

template<char ...Dims, class Struct, class Offset>
constexpr auto symmetric_slice(Struct structure, Offset offset) noexcept {
	return symmetric_slice<Dims...>(structure, empty_state, offset);
}

template<char ...Dims, class Struct, class ...Offsets, class ...StateItems, class = std::enable_if_t<sizeof...(Dims) == sizeof...(Offsets)>>
constexpr auto symmetric_slices(Struct structure, state<StateItems...> state, Offsets ...offsets) noexcept {
	return (... ^ symmetric_slice<Dims>(structure, state, offsets));
}

template<char ...Dims, class Struct, class ...Offsets, class = std::enable_if_t<sizeof...(Dims) == sizeof...(Offsets)>>
constexpr auto symmetric_slices(Struct structure, Offsets ...offsets) noexcept {
	return symmetric_slices<Dims...>(structure, empty_state, offsets...);
}

// State to structure

namespace helpers {

template<char Dim, class IdxT, class State>
constexpr auto state_construct_fix(state_item<index_in<Dim>, IdxT>, State state) {
	return fix<Dim>(state);
}

template<class StateItem, class State>
constexpr auto state_construct_fix(StateItem, State) {
	return neutral_proto();
}

} // namespace helpers

template<class... StateItem>
constexpr auto fix(state<StateItem...> state) noexcept {
	return (neutral_proto() ^ ... ^ helpers::state_construct_fix(StateItem(), state));
}

template<char Dim, char ...Dims, class... StateItem>
constexpr auto fix(state<StateItem...> state) noexcept {
	return fix<Dim, Dims...>(get_index<Dim>(state), get_index<Dims>(state)...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP
