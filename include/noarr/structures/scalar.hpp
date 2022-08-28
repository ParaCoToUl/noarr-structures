#ifndef NOARR_STRUCTURES_SCALAR_HPP
#define NOARR_STRUCTURES_SCALAR_HPP

#include "pipes.hpp"
#include "struct_decls.hpp"
#include "contain.hpp"
#include "type.hpp"

namespace noarr {

namespace helpers {

template<class T, class... KS>
struct scalar_get_t;

template<class T>
struct scalar_get_t<T> {
	using type = T;
};

template<class T>
struct scalar_get_t<T, void> {
	using type = T;
};

}

/**
 * @brief The ground structure for stored data
 * 
 * @tparam T the stored type
 */
template<class T>
struct scalar : contain<> {
	static constexpr std::tuple<> sub_structures() noexcept { return {}; }
	using description = struct_description<
		char_pack<'s', 'c', 'a', 'l', 'a', 'r'>,
		dims_impl<>,
		dims_impl<>,
		type_param<T>>;

	template<class... KS>
	using get_t = typename helpers::scalar_get_t<T, KS...>::type;

	constexpr scalar() noexcept = default;
	static constexpr auto construct() noexcept {
		return scalar<T>();
	}
	static constexpr std::size_t size() noexcept { return sizeof(T); }
	static constexpr std::size_t offset() noexcept { return 0; }
	static constexpr std::size_t length() noexcept { return 0; }

	using struct_type = scalar_type<T>;

	template<class State>
	constexpr std::size_t size(State) const noexcept {
		static_assert(State::is_empty, "Unused items in state");
		return sizeof(T);
	}

	template<class Sub, class State>
	std::size_t strict_offset_of(State) const noexcept {
		static_assert(always_false<Sub>, "Substructure was not found");
		std::terminate();
	}

	template<char QDim, class State>
	std::size_t length(State state) const noexcept {
		static_assert(always_false_dim<QDim>, "Index in this dimension is not accepted by any substructure");
		std::terminate();
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_SCALAR_HPP
