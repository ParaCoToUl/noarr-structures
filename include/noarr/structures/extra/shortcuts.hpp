#ifndef NOARR_STRUCTURES_SHORTCUTS_HPP
#define NOARR_STRUCTURES_SHORTCUTS_HPP

#include "../base/utility.hpp"
#include "../extra/funcs.hpp"
#include "../structs/bcast.hpp"
#include "../structs/layouts.hpp"
#include "../structs/blocks.hpp"
#include "../structs/views.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"

namespace noarr {

// Common compositions

// TODO add tests
template<IsDim auto ... Dims> requires (... && IsDim<decltype(Dims)>)
constexpr auto vectors() noexcept {
	return (... ^ vector<Dims>());
}

template<IsDim auto Dim, class LenT>
constexpr auto sized_vector(LenT length) noexcept {
	return vector<Dim>() ^ set_length<Dim>(length);
}

// TODO add tests
template<auto ...Dims, class ...LenT> requires (... && IsDim<decltype(Dims)>)
constexpr auto sized_vectors(LenT ...lengths) noexcept {
	return (... ^ sized_vector<Dims>(lengths));
}

template<IsDim auto Dim, std::size_t L>
struct array_proto {
	static constexpr bool proto_preserves_layout = false;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return s ^ sized_vector<Dim>(lit<L>); }
};

template<IsDim auto Dim, std::size_t L, class SubStruct>
using array_t = decltype(std::declval<SubStruct>() ^ array_proto<Dim, L>());

template<IsDim auto Dim, std::size_t L>
constexpr auto array() noexcept {
	return array_proto<Dim, L>();
}

template<IsDim auto Dim, class Struct>
constexpr auto length_like(Struct structure, IsState auto state) noexcept {
	return set_length<Dim>(structure | get_length<Dim>(state));
}

template<IsDim auto Dim, class Struct>
constexpr auto length_like(Struct structure) noexcept {
	return length_like<Dim>(structure, empty_state);
}

template<auto ...Dims, class Struct> requires (... && IsDim<decltype(Dims)>)
constexpr auto lengths_like(Struct structure, IsState auto state) noexcept {
	return (... ^ length_like<Dims>(structure, state));
}

template<auto ...Dims, class Struct> requires (... && IsDim<decltype(Dims)>)
constexpr auto lengths_like(Struct structure) noexcept {
	return lengths_like<Dims...>(structure, empty_state);
}

template<IsDim auto Dim, class Struct>
constexpr auto vector_like(Struct structure, IsState auto state) noexcept {
	return (vector<Dim>() ^ length_like<Dim>(structure, state));
}

template<IsDim auto Dim, class Struct>
constexpr auto vector_like(Struct structure) noexcept {
	return vector_like<Dim>(structure, empty_state);
}

template<auto ...Dims, class Struct> requires (... && IsDim<decltype(Dims)>)
constexpr auto vectors_like(Struct structure, IsState auto state) noexcept {
	return (... ^ (vector_like<Dims>(structure, state)));
}

template<auto ...Dims, class Struct> requires (... && IsDim<decltype(Dims)>)
constexpr auto vectors_like(Struct structure) noexcept {
	return vectors_like<Dims...>(structure, empty_state);
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class MinorLengthT>
constexpr auto into_blocks(MinorLengthT minor_length) noexcept {
	return into_blocks<Dim, DimMajor, DimMinor>() ^ set_length<DimMinor>(minor_length);
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class MinorLengthT>
constexpr auto into_blocks_dynamic(MinorLengthT minor_length) noexcept {
	return into_blocks_dynamic<Dim, DimMajor, DimMinor, DimIsPresent>() ^ set_length<DimMinor>(minor_length);
}

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim, class MinorLenT>
constexpr auto merge_blocks(MinorLenT minor_length) noexcept {
	return set_length<DimMinor>(minor_length) ^ merge_blocks<DimMajor, DimMinor, Dim>();
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class... OptionalMinorLengthT>
constexpr auto strip_mine(OptionalMinorLengthT... optional_minor_length) noexcept {
	return into_blocks<Dim, DimMajor, DimMinor>(optional_minor_length...) ^ hoist<DimMajor>();
}

template<auto ...Dims, class ...LenTs> requires (... && IsDim<decltype(Dims)>)
constexpr auto bcast(LenTs ...lengths) noexcept {
	return (... ^ (bcast<Dims>() ^ set_length<Dims>(lengths)));
}

// Working with state (especially in traverser lambdas)

template<IsDim auto Dim>
constexpr auto get_index(IsState auto state) noexcept {
	return state.template get<index_in<Dim>>();
}

template<auto... Dim> requires (... && IsDim<decltype(Dim)>)
constexpr auto get_indices(IsState auto state) noexcept {
	return std::make_tuple(state.template get<index_in<Dim>>()...);
}

template<auto... Dim, class... ValueType> requires (... && IsDim<decltype(Dim)>)
constexpr auto idx(ValueType... value) noexcept {
	return state<state_item<index_in<Dim>, good_index_t<ValueType>>...>(value...);
}

template<IsDim auto Dim, class F>
constexpr auto update_index(IsState auto state, F f) noexcept {
	static_assert(decltype(state)::template contains<index_in<Dim>>, "Requested dimension does not exist. To add a new dimension instead of updating existing one, use .template with<index_in<'...'>>(...)");
	auto new_index = f(state.template get<index_in<Dim>>());
	return state.template with<index_in<Dim>>(good_index_t<decltype(new_index)>(new_index));
}

template<auto... Dims, class ...Diffs> requires ((sizeof...(Dims) == sizeof...(Diffs)) && ... && IsDim<decltype(Dims)>)
constexpr auto neighbor(IsState auto state, Diffs... diffs) noexcept {
	using namespace noarr::constexpr_arithmetic;
	static_assert((... && decltype(state)::template contains<index_in<Dims>>), "Requested dimension does not exist");
	static_assert((... && std::is_same_v<state_get_t<decltype(state), index_in<Dims>>, std::size_t>), "Cannot shift in a dimension that is not dynamic");
	return state.template with<index_in<Dims>...>(good_diff_index_t<decltype(state.template get<index_in<Dims>>() + diffs)>(state.template get<index_in<Dims>>() + diffs)...);
}

// TODO add tests
template<auto ...Dims, class Struct, class Offset, class ...StateItems> requires (... && IsDim<decltype(Dims)>)
constexpr auto symmetric_span(Struct structure, state<StateItems...> state, Offset offset) noexcept {
	return (... ^ span<Dims>(offset, (structure | get_length<Dims>(state)) - offset));
}

// TODO add tests
template<auto ...Dims, class Struct, class Offset> requires (... && IsDim<decltype(Dims)>)
constexpr auto symmetric_span(Struct structure, Offset offset) noexcept {
	return symmetric_span<Dims...>(structure, empty_state, offset);
}

// TODO add tests
template<auto ...Dims, class Struct, class ...Offsets, class ...StateItems> requires ((sizeof...(Dims) == sizeof...(Offsets)) &&  ... && IsDim<decltype(Dims)>)
constexpr auto symmetric_spans(Struct structure, state<StateItems...> state, Offsets ...offsets) noexcept {
	return (... ^ symmetric_span<Dims>(structure, state, offsets));
}

// TODO add tests
template<auto ...Dims, class Struct, class ...Offsets> requires ((sizeof...(Dims) == sizeof...(Offsets)) &&  ... && IsDim<decltype(Dims)>)
constexpr auto symmetric_spans(Struct structure, Offsets ...offsets) noexcept {
	return symmetric_spans<Dims...>(structure, empty_state, offsets...);
}

// State to structure

namespace helpers {

template<IsDim auto Dim, class IdxT>
constexpr auto state_construct_fix(state_item<index_in<Dim>, IdxT>, IsState auto state) noexcept {
	return fix<Dim>(state);
}

template<class StateItem>
constexpr auto state_construct_fix(StateItem, IsState auto) noexcept {
	return neutral_proto();
}

} // namespace helpers

template<class... StateItem>
constexpr auto fix(state<StateItem...> state) noexcept {
	(void)state;
	return (neutral_proto() ^ ... ^ helpers::state_construct_fix(StateItem(), state));
}

template<IsDim auto Dim, auto ...Dims, class... StateItem> requires (... && IsDim<decltype(Dims)>)
constexpr auto fix(state<StateItem...> state) noexcept {
	return fix<Dim, Dims...>(get_index<Dim>(state), get_index<Dims>(state)...);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SHORTCUTS_HPP
