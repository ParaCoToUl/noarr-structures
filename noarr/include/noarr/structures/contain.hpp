#ifndef NOARR_STRUCTURES_CONTAIN_HPP
#define NOARR_STRUCTURES_CONTAIN_HPP

#include <type_traits>

// TODO: write a better get<..>

namespace noarr {

template<typename, typename... TS>
struct _contain;

template<typename... TS>
using contain = _contain<void, TS...>;

template<typename T, std::size_t I>
struct _contain_get {
    static constexpr auto get(const T &t) {
        return t.template _get_next<I>();
    }
};

template<typename T>
struct _contain_get<T, 0> {
    static constexpr auto get(const T &t) {
        return t._get();
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<!std::is_empty<T>::value && !std::is_empty<contain<TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> {
    typename<typename, typename...>
    friend struct _contain;

    T t;
    contain<TS...> ts;

    constexpr _contain() = default;
    explicit constexpr _contain(T t, TS... ts) : t(t), ts(ts...) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

private:
    template<std::size_t I>
    constexpr auto _get_next() const {
        return ts.template get<I - 1>();
    }

    constexpr auto _get() const {
        return t;
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<!std::is_empty<T>::value && std::is_empty<contain<TS...>>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain<TS...> {
    typename<typename, typename...>
    friend struct _contain;

    T t;

    constexpr _contain() = default;
    explicit constexpr _contain(T t) : t(t) {}
    explicit constexpr _contain(T t, TS...) : t(t) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

private:
    template<std::size_t I>
    constexpr auto _get_next() const {
        return contain<TS...>::template get<I - 1>();
    }

    constexpr auto _get() const {
        return t;
    }
};

template<typename T, typename... TS>
struct _contain<std::enable_if_t<std::is_empty<T>::value && (sizeof...(TS) > 0)>, T, TS...> : private contain<TS...> {
    typename<typename, typename...>
    friend struct _contain;

    constexpr _contain() = default;
    explicit constexpr _contain(TS... ts) : contain<TS...>(ts...) {}
    explicit constexpr _contain(T, TS... ts) : contain<TS...>(ts...) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

private:
    template<std::size_t I>
    constexpr auto _get_next() const {
        return contain<TS...>::template get<I - 1>();
    }

    constexpr auto _get() const {
        return T();
    }
};

template<typename T>
struct _contain<std::enable_if_t<std::is_empty<T>::value>, T> {
    typename<typename, typename...>
    friend struct _contain;

    constexpr _contain() = default;
    explicit constexpr _contain(T) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

private:
    constexpr auto _get() const {
        return T();
    }
};

template<typename T>
struct _contain<std::enable_if_t<!std::is_empty<T>::value>, T> {
    typename<typename, typename...>
    friend struct _contain;

    T t;

    constexpr _contain() = default;
    explicit constexpr _contain(T t) : t(t) {}

    template<std::size_t I>
    constexpr auto get() const {
        return _contain_get<_contain, I>::get(*this);
    }

private:
    constexpr auto _get() const {
        return t;
    }
};

template<>
struct _contain<void> {};

}

#endif // NOARR_STRUCTURES_CONTAIN_HPP