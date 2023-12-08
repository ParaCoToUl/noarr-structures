#ifndef NOARR_STRUCTURES_CONTAIN_HPP
#define NOARR_STRUCTURES_CONTAIN_HPP

#include <cstddef>
#include <type_traits>
#include <tuple>
#include <utility>

#include "utility.hpp"

namespace noarr {

namespace helpers {

/**
 * @brief see `contain`. This is a helper structure implementing its functionality
 *
 * @tparam ...TS: contained fields
 */
template<class... TS>
struct contain_impl;

// an implementation for the pair (T, TS...) where neither is empty
template<class T, class... TS> requires (!std::is_empty_v<T> && !std::is_empty_v<contain_impl<TS...>>)
struct contain_impl<T, TS...> {
	template<class T_, class ...TS_> requires (sizeof...(TS) == sizeof...(TS_))
	explicit constexpr contain_impl(T_ &&t, TS_ &&...ts) noexcept : t_(std::forward<T_>(t)), ts_(std::forward<TS_>(ts)...) {}

	template<std::size_t I> requires (I < 1 + sizeof...(TS))
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return ts_.template get<I - 1>();
	}

private:
	contain_impl<T> t_;
	contain_impl<TS...> ts_;
};

// an implementation for the pair (T, TS...) where TS... is empty
template<class T, class... TS> requires (!std::is_empty_v<T> && std::is_empty_v<contain_impl<TS...>>)
struct contain_impl<T, TS...> : private contain_impl<TS...> {
	template<class T_, class ...TS_> requires (sizeof...(TS) == sizeof...(TS_))
	explicit constexpr contain_impl(T_ &&t, const TS_ &...) noexcept : t_(std::forward<T_>(t)) {}

	template<std::size_t I> requires (I < 1 + sizeof...(TS))
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return contain_impl<TS...>::template get<I - 1>();
	}

private:
	contain_impl<T> t_;
};

// an implementation for the pair (T, TS...) where T is empty
template<class T, class... TS> requires (std::is_empty_v<T>)
struct contain_impl<T, TS...> : private contain_impl<TS...> {
	constexpr contain_impl() noexcept = default;
	template<class T_, class ...TS_> requires (sizeof...(TS) == sizeof...(TS_))
	explicit constexpr contain_impl(const T_ &, TS_ &&...ts) noexcept : contain_impl<TS...>(std::forward<TS_>(ts)...) {}

	template<std::size_t I> requires (I < 1 + sizeof...(TS))
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return T();
		else
			return contain_impl<TS...>::template get<I - 1>();
	}
};

// an implementation for an empty T
template<class T> requires (std::is_empty_v<T>)
struct contain_impl<T> {
	constexpr contain_impl() noexcept = default;
	template<class T_> requires (!std::is_base_of_v<contain_impl, std::remove_cvref_t<T_>>)
	explicit constexpr contain_impl(const T_ &) noexcept {}

	template<std::size_t I = 0> requires (I == 0)
	constexpr decltype(auto) get() const noexcept {
		return T();
	}
};

// an implementation for an nonempty T
template<class T> requires (!std::is_empty_v<T>)
struct contain_impl<T> {
	constexpr contain_impl() noexcept = delete;
	template<class T_> requires (!std::is_base_of_v<contain_impl, std::remove_cvref_t<T_>>)
	explicit constexpr contain_impl(T_ &&t) noexcept : t_(std::forward<T_>(t)) {}

	template<std::size_t I = 0> requires (I == 0)
	constexpr const T &get() const noexcept {
		return t_;
	}

private:
	T t_;
};

// contains nothing
template<>
struct contain_impl<> {};

template<class... TS>
struct contain : private helpers::contain_impl<TS...> {
	using helpers::contain_impl<TS...>::contain_impl;
	using helpers::contain_impl<TS...>::get;
};

template<>
struct contain<> : private helpers::contain_impl<> {
	using helpers::contain_impl<>::contain_impl;
};

template<class... TS>
contain(TS &&...) -> contain<std::remove_cvref_t<TS>...>;

template<class... TS1, class... TS2, class... Contains>
constexpr auto contain_cat(const contain<TS1...> &c1, const contain<TS2...> &c2, Contains &&...contains) noexcept {
	using iss1 = std::index_sequence_for<TS1...>;
	using iss2 = std::index_sequence_for<TS2...>;
	return contain_cat(c1, iss1(), c2, iss2(), std::forward<Contains>(contains)...);
}

template<class... TS1, std::size_t... Idxs1, class... TS2, std::size_t... Idxs2, class... Contains>
constexpr auto contain_cat(const contain<TS1...> &c1, std::index_sequence<Idxs1...>, const contain<TS2...> &c2, std::index_sequence<Idxs2...>, Contains &&...contains) noexcept {
	return contain_cat(contain(c1.template get<Idxs1>()..., c2.template get<Idxs2>()...), std::forward<Contains>(contains)...);
}

constexpr auto contain_cat() noexcept {
	return contain();
}

template<class... TS>
constexpr auto contain_cat(const contain<TS...> &c) noexcept {
	return c;
}

} // namespace helpers

/**
 * @brief A base class that contains the fields given as template arguments. It is similar to a tuple but it is a standard layout.
 *
 * @tparam TS the contained fields
 */
template<class... TS> requires (... && IsSimple<TS>)
using strict_contain = helpers::contain<TS...>;

/**
 * @brief A base class that contains the fields given as template arguments. It is similar to a tuple but it is a standard layout.
 *
 * @tparam TS the contained fields
 */
template<class... TS> requires (... && IsContainable<TS>)
using flexible_contain = helpers::contain<TS...>;

template<class... TS>
constexpr auto contain_cat(TS &&...ts) noexcept {
	return helpers::contain_cat(std::forward<TS>(ts)...);
}

} // namespace noarr

template<std::size_t I, class... TS>
struct std::tuple_element<I, noarr::helpers::contain<TS...>> {
	using type = decltype(std::declval<noarr::helpers::contain<TS...>>().template get<I>());
};

template<class... TS>
struct std::tuple_size<noarr::helpers::contain<TS...>>
	: std::integral_constant<std::size_t, sizeof...(TS)> { };

#endif // NOARR_STRUCTURES_CONTAIN_HPP
