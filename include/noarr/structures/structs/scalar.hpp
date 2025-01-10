#ifndef NOARR_STRUCTURES_SCALAR_HPP
#define NOARR_STRUCTURES_SCALAR_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

/**
 * @brief The ground structure for stored data
 *
 * @tparam T the stored type
 */
template<class T>
struct scalar : strict_contain<> {
	using strict_contain<>::strict_contain;

	static constexpr char name[] = "scalar";
	using params = struct_params<type_param<T>>;

	using signature = scalar_sig<T>;

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return true;
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto size(State /*unused*/ = empty_state) noexcept
	requires (has_size<State>())
	{
		return constexpr_arithmetic::make_const<sizeof(T)>();
	}

	template<IsState State = state<>>
	[[nodiscard]]
	static constexpr auto align(State /*unused*/ = empty_state) noexcept
	requires (has_size<State>())
	{
		return constexpr_arithmetic::make_const<alignof(T)>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return false;
	}

	template<class Sub, IsState State = state<>>
	static constexpr void strict_offset_of(State /*unused*/ = empty_state) noexcept
	requires (has_offset_of<Sub, scalar, State>())
	{}

	template<auto QDim, IsState State>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		return false;
	}

	template<IsDim auto QDim, IsState State = state<>>
	static constexpr void length(State /*unused*/ = empty_state) noexcept
	requires (has_length<QDim, State>())
	{}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return false;
	}

	template<class Sub, IsState State = state<>>
	static constexpr void strict_state_at(State /*unused*/ = empty_state) noexcept
	requires (has_state_at<Sub, scalar, State>())
	{}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_SCALAR_HPP
