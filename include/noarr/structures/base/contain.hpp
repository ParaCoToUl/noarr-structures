#ifndef NOARR_STRUCTURES_CONTAIN_HPP
#define NOARR_STRUCTURES_CONTAIN_HPP

#include <type_traits>
#include <tuple>

#include "utility.hpp"

namespace noarr {

/**
 * @brief A base class that contains the fields given as template arguments. It is similar to a tuple but it is a standard layout.
 * 
 * @tparam TS the contained fields
 */
template<class... TS> requires (... && IsSimple<TS>)
struct contain;

// an implementation for the pair (T, TS...) where neither is empty
template<class T, class... TS> requires (!std::is_empty_v<T> && !std::is_empty_v<contain<TS...>>)
struct contain<T, TS...> {
	explicit constexpr contain(T t, TS... ts) noexcept : t_(t), ts_(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return ts_.template get<I - 1>();
	}

private:
	contain<T> t_;
	contain<TS...> ts_;
};

// an implementation for the pair (T, TS...) where TS... is empty
template<class T, class... TS> requires (!std::is_empty_v<T> && std::is_empty_v<contain<TS...>>)
struct contain<T, TS...> : private contain<TS...> {
	explicit constexpr contain(T t, TS...) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return contain<TS...>::template get<I - 1>();
	}

private:
	contain<T> t_;
};

// an implementation for the pair (T, TS...) where T is empty
template<class T, class... TS> requires (std::is_empty_v<T>)
struct contain<T, TS...> : private contain<TS...> {
	constexpr contain() noexcept = default;
	explicit constexpr contain(T, TS... ts) noexcept : contain<TS...>(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return T();
		else
			return contain<TS...>::template get<I - 1>();
	}
};

// an implementation for an empty T
template<class T> requires (std::is_empty_v<T>)
struct contain<T> {
	constexpr contain() noexcept = default;
	explicit constexpr contain(T) noexcept {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		static_assert(I == 0, "index out of bounds");
		return T();
	}
};

// an implementation for an nonempty T
template<class T> requires (!std::is_empty_v<T>)
struct contain<T> {
	constexpr contain() noexcept = delete;
	explicit constexpr contain(T t) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		static_assert(I == 0, "index out of bounds");
		return t_;
	}

private:
	T t_;
};

// contains nothing
template<>
struct contain<> {};

} // namespace noarr


namespace std {

template<std::size_t I, class... TS>
struct tuple_element<I, noarr::contain<TS...>> {
	using type = decltype(std::declval<noarr::contain<TS...>>().template get<I>());
};

template<class... TS>
struct tuple_size<noarr::contain<TS...>>
	: std::integral_constant<std::size_t, sizeof...(TS)> { };

} // namespace std

#endif // NOARR_STRUCTURES_CONTAIN_HPP
