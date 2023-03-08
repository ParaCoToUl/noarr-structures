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

/**
 * @brief structure that facilitates `.get` method of the `contain` structure
 * 
 * @tparam T: the `contain` structure
 * @tparam I: the index of the desired field
 */
template<class T, std::size_t I>
struct contain_get {
	static constexpr decltype(auto) get(const T &t) noexcept {
		return t.template _get_next<I>();
	}

	static constexpr decltype(auto) get(T &t) noexcept {
		return t.template _get_next<I>();
	}
};

template<class T>
struct contain_get<T, 0> {
	static constexpr decltype(auto) get(const T &t) noexcept {
		return t._get();
	}

	static constexpr decltype(auto) get(T &t) noexcept {
		return t._get();
	}
};

// an implementation for the pair (T, TS...) where neither is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value && !std::is_empty<contain_impl<void, TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> {
	template<class, std::size_t>
	friend struct contain_get;

	T t_;
	contain_impl<void, TS...> ts_;

	constexpr contain_impl() noexcept = default;
	explicit constexpr contain_impl(T t, TS... ts) noexcept : t_(t), ts_(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

	template<std::size_t I>
	constexpr decltype(auto) get() noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

private:
	template<std::size_t I>
	constexpr decltype(auto) _get_next() const noexcept {
		return ts_.template get<I - 1>();
	}

	template<std::size_t I>
	constexpr decltype(auto) _get_next() noexcept {
		return ts_.template get<I - 1>();
	}

	constexpr const auto &_get() const noexcept {
		return t_;
	}

	constexpr auto &_get() noexcept {
		return t_;
	}
};

// an implementation for the pair (T, TS...) where TS... is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value && std::is_empty<contain_impl<void, TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain_impl<void, TS...> {
	template<class, std::size_t>
	friend struct contain_get;

	T t_;

	constexpr contain_impl() noexcept = default;
	explicit constexpr contain_impl(T t) noexcept : t_(t) {}
	explicit constexpr contain_impl(T t, TS...) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

	template<std::size_t I>
	constexpr decltype(auto) get() noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

private:
	template<std::size_t I>
	constexpr decltype(auto) _get_next() const noexcept {
		return contain_impl<void, TS...>::template get<I - 1>();
	}

	template<std::size_t I>
	constexpr decltype(auto) _get_next() noexcept {
		return contain_impl<void, TS...>::template get<I - 1>();
	}

	constexpr const auto &_get() const noexcept {
		return t_;
	}

	constexpr auto &_get() noexcept {
		return t_;
	}
};

// an implementation for the pair (T, TS...) where T is empty
template<class T, class... TS>
struct contain_impl<std::enable_if_t<std::is_empty<T>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain_impl<void, TS...> {
	template<class, std::size_t>
	friend struct contain_get;

	constexpr contain_impl() noexcept = default;
	explicit constexpr contain_impl(TS... ts) noexcept : contain_impl<void, TS...>(ts...) {}
	explicit constexpr contain_impl(T, TS... ts) noexcept : contain_impl<void, TS...>(ts...) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

	template<std::size_t I>
	constexpr decltype(auto) get() noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

private:
	template<std::size_t I>
	constexpr decltype(auto) _get_next() const noexcept {
		return contain_impl<void, TS...>::template get<I - 1>();
	}

	template<std::size_t I>
	constexpr decltype(auto) _get_next() noexcept {
		return contain_impl<void, TS...>::template get<I - 1>();
	}

	static constexpr auto _get() noexcept {
		return T();
	}
};

// an implementation for an empty T
template<class T>
struct contain_impl<std::enable_if_t<std::is_empty<T>::value>, T> {
	template<class, std::size_t>
	friend struct contain_get;

	constexpr contain_impl() noexcept = default;
	explicit constexpr contain_impl(T) noexcept {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

	template<std::size_t I>
	constexpr decltype(auto) get() noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

private:
	static constexpr auto _get() noexcept {
		return T();
	}
};

// an implementation for an nonempty T
template<class T>
struct contain_impl<std::enable_if_t<!std::is_empty<T>::value>, T> {
	template<class, std::size_t>
	friend struct contain_get;

	T t_;

	constexpr contain_impl() noexcept = delete;
	explicit constexpr contain_impl(T t) noexcept : t_(t) {}

	template<std::size_t I>
	constexpr decltype(auto) get() const noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

	template<std::size_t I>
	constexpr decltype(auto) get() noexcept {
		return contain_get<contain_impl, I>::get(*this);
	}

private:
	constexpr const auto &_get() const noexcept {
		return t_;
	}

	constexpr auto &_get() noexcept {
		return t_;
	}
};

// contains nothing
template<>
struct contain_impl<void> {};

template<class... TS>
struct contain_wrapper : private contain_impl<void, TS...> {
protected:
	using contain_impl<void, TS...>::contain_impl;

public:
	using contain_impl<void, TS...>::get;
};

template<>
struct contain_wrapper<> : private contain_impl<void> {
protected:
	using contain_impl<void>::contain_impl;
};

} // namespace helpers

/**
 * @brief A base class that contains the fields given as template arguments. It is similar to a tuple but it is a standard layout.
 * 
 * @tparam TS the contained fields
 */
template<class... TS>
using contain = helpers::contain_wrapper<TS...>;

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
