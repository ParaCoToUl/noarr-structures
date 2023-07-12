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
struct scalar : contain<> {
	static constexpr char name[] = "scalar";
	using params = struct_params<
		type_param<T>>;

	constexpr scalar() noexcept = default;

	using signature = scalar_sig<T>;

	static constexpr auto size(IsState auto) noexcept {
		return constexpr_arithmetic::make_const<sizeof(T)>();
	}

	template<class Sub>
	static constexpr void strict_offset_of(IsState auto) noexcept {
		static_assert(always_false<Sub>, "Substructure was not found");
	}

	template<IsDim auto QDim>
	static constexpr void length(IsState auto) noexcept {
		static_assert(value_always_false<QDim>, "Index in this dimension is not accepted by any substructure");
	}

	template<class Sub>
	static constexpr void strict_state_at(IsState auto) noexcept {
		static_assert(always_false<scalar<T>>, "A scalar cannot be used in this context");
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_SCALAR_HPP
