#ifndef NOARR_STRUCTURES_SCALAR_HPP
#define NOARR_STRUCTURES_SCALAR_HPP

#include "struct_decls.hpp"
#include "contain.hpp"
#include "signature.hpp"

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
	static constexpr std::size_t size() noexcept { return sizeof(T); }
	static constexpr std::size_t offset() noexcept { return 0; }
	static constexpr std::size_t length() noexcept { return 0; }

	using signature = scalar_sig<T>;

	template<class State>
	constexpr std::size_t size(State) const noexcept {
		static_assert(State::is_empty, "Unused items in state");
		return sizeof(T);
	}

	template<class Sub, class State>
	constexpr void strict_offset_of(State) const noexcept {
		static_assert(always_false<Sub>, "Substructure was not found");
	}

	template<char QDim, class State>
	constexpr void length(State) const noexcept {
		static_assert(always_false_dim<QDim>, "Index in this dimension is not accepted by any substructure");
	}

	template<class Sub, class State>
	constexpr void strict_state_at(State) const noexcept {
		static_assert(always_false<scalar<T>>, "A scalar cannot be used in this context");
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_SCALAR_HPP
