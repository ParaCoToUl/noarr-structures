#ifndef NOARR_TEST_DEFS_HPP
#define NOARR_TEST_DEFS_HPP

#include <type_traits>

namespace noarr_test {
	// A trivial-enough type. In addition to being a standard-layout type, it must satisfy the following conditions:
	// - All special member functions must be trivial, if present.
	// - Copy ctor, move ctor, dtor must be present.
	// - Default ctor and assignments are optional.
	// - Default ctor must *not* be present for nonempty structures.
	template<class T>
	static constexpr bool is_simple = true
		&& std::is_standard_layout_v<T>
		&& (std::is_trivially_default_constructible_v<T> ? std::is_empty_v<T> : !std::is_default_constructible_v<T>)
		&& std::is_trivially_copy_constructible_v<T>
		&& std::is_trivially_move_constructible_v<T>
		&& (std::is_trivially_copy_assignable_v<T> || !std::is_copy_assignable_v<T>)
		&& (std::is_trivially_move_assignable_v<T> || !std::is_move_assignable_v<T>)
		&& std::is_trivially_destructible_v<T>
		;

	template<class T>
	static constexpr bool type_is_simple(const T&) {
		return is_simple<T>;
	}
}

#endif // NOARR_TEST_DEFS_HPP
