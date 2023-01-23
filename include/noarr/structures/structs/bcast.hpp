#ifndef NOARR_STRUCTURES_BCAST_HPP
#define NOARR_STRUCTURES_BCAST_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<char Dim, class T>
struct bcast_t : contain<T> {
	static constexpr char name[] = "bcast_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	constexpr bcast_t() noexcept = default;
	explicit constexpr bcast_t(T sub_structure) noexcept : contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return contain<T>::template get<0>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, unknown_arg_length, typename T::signature>;

	template<class State>
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(state.template remove<index_in<Dim>, length_in<Dim>>());
	}

	template<class Sub, class State>
	constexpr auto strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		return offset_of<Sub>(sub_structure(), state.template remove<index_in<Dim>, length_in<Dim>>());
	}

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			static_assert(State::template contains<length_in<Dim>>, "This length has not been set yet");
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(state.template remove<index_in<Dim>, length_in<Dim>>());
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		return state_at<Sub>(sub_structure(), state.template remove<index_in<Dim>, length_in<Dim>>());
	}
};

template<char Dim>
struct bcast_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return bcast_t<Dim, Struct>(s); }
};

template<char... Dim>
constexpr auto bcast() noexcept { return (... ^ bcast_proto<Dim>()); }

template<>
constexpr auto bcast<>() noexcept { return neutral_proto(); }

} // namespace noarr

#endif // NOARR_STRUCTURES_BCAST_HPP
