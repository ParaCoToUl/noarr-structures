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
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "bcast_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>>;

	template<IsState State>
	constexpr T sub_structure(State /*state*/) const noexcept {
		return strict_contain<T>::get();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return strict_contain<T>::get();
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto sub_state(State state) noexcept {
		return state.template remove<index_in<Dim>, length_in<Dim>>();
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<index_in<Dim>, length_in<Dim>>();
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(sub_state(std::declval<State>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, dynamic_arg_length, typename T::signature>;

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto align(State state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (QDim == Dim) {
			return state_contains<State, length_in<Dim>> && !state_contains<State, index_in<Dim>>;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		if constexpr (QDim == Dim) {
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}
};

template<IsDim auto Dim>
struct bcast_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return bcast_t<Dim, Struct>(s);
	}
};

template<auto... Dims>
requires IsDimPack<decltype(Dims)...>
constexpr auto bcast() noexcept {
	return (... ^ bcast_proto<Dims>());
}

template<>
constexpr auto bcast<>() noexcept {
	return neutral_proto();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_BCAST_HPP
