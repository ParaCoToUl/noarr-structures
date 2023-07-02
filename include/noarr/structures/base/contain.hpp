#ifndef NOARR_STRUCTURES_CONTAIN_HPP
#define NOARR_STRUCTURES_CONTAIN_HPP

#include <type_traits>
#include <tuple>

#include "utility.hpp"

namespace noarr {

namespace helpers {

/**
 * @brief see `contain`. This is a helper structure implementing its functionality
 * 
 * @tparam class: placeholder type
 * @tparam ...TS: contained fields
 */
template<class, class... TS>
struct contain_impl;

// an implementation for the pair (T, TS...) where neither is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value && !std::is_empty<contain_impl<void, TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> {
	contain_impl<void, T> t_;
	contain_impl<void, TS...> ts_;

	explicit constexpr contain_impl(T t, TS... ts) noexcept : t_(t), ts_(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return ts_.template get<I - 1>();
	}
};

// an implementation for the pair (T, TS...) where TS... is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value && std::is_empty<contain_impl<void, TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain_impl<void, TS...> {
	contain_impl<void, T> t_;

	explicit constexpr contain_impl(T t, TS...) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return t_.template get<0>();
		else
			return contain_impl<void, TS...>::template get<I - 1>();
	}
};

// an implementation for the pair (T, TS...) where T is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<std::is_empty<T>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain_impl<void, TS...> {
	constexpr contain_impl() = default;
	explicit constexpr contain_impl(T, TS... ts) noexcept : contain_impl<void, TS...>(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		if constexpr(I == 0)
			return T();
		else
			return contain_impl<void, TS...>::template get<I - 1>();
	}
};

// an implementation for an empty T
template<class T>
struct contain_impl<std::enable_if_t<std::is_empty<T>::value>, T> {
	constexpr contain_impl() noexcept = default;
	explicit constexpr contain_impl(T) noexcept {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		static_assert(I == 0, "index out of bounds");
		return T();
	}
};

// an implementation for an nonempty T
template<class T>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value>, T> {
	T t_;

	constexpr contain_impl() noexcept = delete;
	explicit constexpr contain_impl(T t) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		static_assert(I == 0, "index out of bounds");
		return t_;
	}
};

// contains nothing
template<>
struct contain_impl<void> {};

} // namespace helpers

/**
 * @brief A base class that contains the fields given as template arguments. It is similar to a tuple but it is a standard layout.
 * 
 * @tparam TS the contained fields
 */
template<class... TS>
struct contain : private helpers::contain_impl<void, TS...> {
protected:
	using helpers::contain_impl<void, TS...>::contain_impl;

public:
	using helpers::contain_impl<void, TS...>::get;
};

template<>
struct contain<> : private helpers::contain_impl<void> {
protected:
	using helpers::contain_impl<void>::contain_impl;
};

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
