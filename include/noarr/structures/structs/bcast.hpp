#ifndef NOARR_STRUCTURES_BCAST_HPP
#define NOARR_STRUCTURES_BCAST_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<IsDim auto Dim, class T>
struct bcast_t : strict_contain<T> {
	static constexpr char name[] = "bcast_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	constexpr bcast_t() noexcept = default;
	explicit constexpr bcast_t(T sub_structure) noexcept : strict_contain<T>(sub_structure) {}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return strict_contain<T>::get(); }

	[[nodiscard]]
	constexpr auto sub_state(IsState auto state) const noexcept { return state.template remove<index_in<Dim>, length_in<Dim>>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, unknown_arg_length, typename T::signature>;

	[[nodiscard]]
	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	[[nodiscard]]
	constexpr auto align(IsState auto state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State> requires (HasSetIndex<State, Dim>)
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State> requires (QDim != Dim || HasNotSetIndex<State, QDim>) && IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		if constexpr(QDim == Dim) {
			static_assert(State::template contains<length_in<Dim>>, "This length has not been set yet");
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim>
struct bcast_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return bcast_t<Dim, Struct>(s); }
};

template<auto ...Dims> requires IsDimPack<decltype(Dims)...>
constexpr auto bcast() noexcept { return (... ^ bcast_proto<Dims>()); }

template<>
constexpr auto bcast<>() noexcept { return neutral_proto(); }

} // namespace noarr

#endif // NOARR_STRUCTURES_BCAST_HPP
