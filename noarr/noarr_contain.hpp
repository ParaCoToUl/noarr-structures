#ifndef NOARR_CONTAIN_HPP
#define NOARR_CONTAIN_HPP

#include <type_traits>

namespace noarr {

template<typename, typename... TS>
struct _contain;

template<typename... TS>
using contain = _contain<void, TS...>;

template<typename T, std::size_t I>
struct _contain_get {
    static constexpr auto get(const T &t) {
        return t.template get_next_<I>();
    }
};

template<typename T>
struct _contain_get<T, 0> {
    static constexpr auto get(const T &t) {
        return t.get_();
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<!std::is_empty<T>::value && !std::is_empty<contain<TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> {
    T t;
    contain<TS...> ts;

    constexpr _contain() = default;
    explicit constexpr _contain(T t, TS... ts) : t(t), ts(ts...) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

    template<std::size_t I>
    constexpr auto get_next_() const {
        return ts.template get<I - 1>();
    }

    constexpr auto get_() const {
        return t;
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<!std::is_empty<T>::value && std::is_empty<contain<TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain<TS...> {
    T t;

    constexpr _contain() = default;
    explicit constexpr _contain(T t) : t(t) {}
    explicit constexpr _contain(T t, TS...) : t(t) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

    template<std::size_t I>
    constexpr auto get_next_() const {
        return contain<TS...>::template get<I - 1>();
    }

    constexpr auto get_() const {
        return t;
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<std::is_empty<T>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain<TS...> {
    constexpr _contain() = default;
    explicit constexpr _contain(TS... ts) : contain<TS...>(ts...) {}
    explicit constexpr _contain(T, TS... ts) : contain<TS...>(ts...) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

    template<std::size_t I>
    constexpr auto get_next_() const {
        return contain<TS...>::template get<I - 1>();
    }

    constexpr auto get_() const {
        return T();
    }
};

template<typename T>
struct _contain<std::enable_if_t<std::is_empty<T>::value>, T> {
    constexpr _contain() = default;
    explicit constexpr _contain(T) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

    constexpr auto get_() const {
        return T();
    }
};

template<typename T>
struct _contain<std::enable_if_t<!std::is_empty<T>::value>, T> {
    T t;

    constexpr _contain() = default;
    explicit constexpr _contain(T t) : t(t) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

    constexpr auto get_() const {
        return t;
    }
};

template<>
struct _contain<void> {};

}

#endif // NOARR_CONTAIN_HPP