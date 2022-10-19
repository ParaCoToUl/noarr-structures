#ifndef NOARR_DIMENSION_MAP_HPP
#define NOARR_DIMENSION_MAP_HPP

#include "utility.hpp"
#include "contain.hpp"

namespace noarr {

namespace helpers {

template<char Dim, class Idx>
struct index_pair {
    using type = Idx;
    static constexpr char dim = Dim;
};

template<class ...IndexPairs>
struct dimension_map;


template<char QDim, bool Const, class Test, class ...IndexPairs>
struct dimension_map_get_t;

template<char QDim, bool Const, char Dim, class Idx, class ...IndexPairs>
struct dimension_map_get_t<QDim, Const, std::enable_if_t<Dim != QDim>, index_pair<Dim, Idx>, IndexPairs...> : public dimension_map_get_t<QDim, Const, void, IndexPairs...> {
};

template<char Dim, bool Const, class Idx, class ...IndexPairs>
struct dimension_map_get_t<Dim, Const, void, index_pair<Dim, Idx>, IndexPairs...> {
private:
    using Container = contain<Idx, dimension_map<IndexPairs...>>;

public:
    using type = decltype(std::declval<std::conditional_t<Const, const Container *, Container *>>()->template get<0>());
};

template<char Dim, class Idx, class ...IndexPairs>
struct dimension_map<index_pair<Dim, Idx>, IndexPairs...> : private contain<Idx, dimension_map<IndexPairs...>> {
    using base = contain<Idx, dimension_map<IndexPairs...>>;

    constexpr dimension_map() noexcept = default;

    template<class ...Idxs>
    constexpr dimension_map(Idx idx, Idxs ...idxs) noexcept : base(idx, dimension_map<IndexPairs...>(idxs...)) {}

    template<char QDim, bool Const>
    using ValueType = typename dimension_map_get_t<QDim, Const, void, index_pair<Dim, Idx>, IndexPairs...>::type;

    template<char QDim>
    constexpr auto get() const noexcept -> std::enable_if_t<QDim == Dim, ValueType<QDim, true>> {
        return base::template get<0>();
    }

    template<char QDim>
    constexpr auto get() noexcept -> std::enable_if_t<QDim == Dim, ValueType<QDim, false>> {
        return base::template get<0>();
    }

    template<char QDim>
    constexpr auto get() const noexcept -> std::enable_if_t<QDim != Dim, ValueType<QDim, true>> {
        return base::template get<1>().template get<QDim>();
    }

    template<char QDim>
    constexpr auto get() noexcept -> std::enable_if_t<QDim != Dim, ValueType<QDim, false>> {
        return base::template get<1>().template get<QDim>();
    }
};

template<>
struct dimension_map<> {
    constexpr dimension_map() noexcept = default;

    template<char QDim>
    using ValueType = void;
};

}

}

#endif // NOARR_DIMENSION_MAP_HPP