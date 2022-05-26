#ifndef NOARR_ZIP_HPP
#define NOARR_ZIP_HPP

#include <tuple>

#include "dimension_map.hpp"
#include "std_ext.hpp"
#include "pipes.hpp"
#include "funcs.hpp"

namespace noarr {

namespace helpers {

struct generator_end_iterator {};

template<class ValueType, class MoveNextFtor>
struct generator_iterator {
    using value_type = ValueType;

private:
    value_type current;
    bool has_next;
    MoveNextFtor move_next;

public:
    explicit constexpr generator_iterator(const value_type &begin, bool non_empty, const MoveNextFtor &move_next) : current(begin), has_next(non_empty), move_next(move_next) {}

    constexpr value_type &operator*() noexcept {
        return current;
    }

    constexpr const value_type &operator*() const noexcept {
        return current;
    }

    generator_iterator &operator++() noexcept {
        has_next = move_next(current);
        return *this;
    }

    constexpr bool operator==(const generator_end_iterator &) const noexcept {
        return !has_next;
    }

    constexpr bool operator!=(const generator_end_iterator &) const noexcept {
        return has_next;
    }
};

template<class ValueType, class MoveNextFtor>
struct generator_range {
    using value_type = ValueType;
    using iterator_type = generator_iterator<value_type, MoveNextFtor>;
    using end_iterator_type = generator_end_iterator;

private:
    value_type begin_value;
    bool non_empty;
    MoveNextFtor move_next;

public:
    explicit constexpr generator_range(const value_type &begin, bool non_empty, const MoveNextFtor &move_next) : begin_value(begin), non_empty(non_empty), move_next(move_next) {}

    constexpr iterator_type begin() const noexcept {
        return iterator_type(begin_value, non_empty, move_next);
    }

    static constexpr end_iterator_type end() noexcept {
        return end_iterator_type();
    }
};

template<class ValueType, class MoveNextFtor>
auto generate(const ValueType &begin, bool non_empty, const MoveNextFtor &move_next) {
    return generator_range<ValueType, MoveNextFtor>(begin, non_empty, move_next);
}

template<char ...Dims>
struct dimension_map_helpers {
private:
    using DimensionMap = dimension_map<index_pair<Dims, std::size_t>...>;

public:
    static constexpr bool increment(DimensionMap &idx, const DimensionMap &lengths) noexcept {
        return ((lengths.template get<Dims>() - idx.template get<Dims>() == 1 ? (idx.template get<Dims>() = 0, false) : (idx.template get<Dims>()++, true)) || ...);
    }

    static constexpr bool equals(const DimensionMap &a, const DimensionMap &b) noexcept {
        return ((a.template get<Dims>() == b.template get<Dims>()) && ...);
    }

    static constexpr bool all_nonzero(const DimensionMap &lengths) noexcept {
        return (lengths.template get<Dims>() && ...);
    }

    template<class Struct>
    static constexpr DimensionMap get_lengths(const Struct &s) noexcept {
        return DimensionMap((s | get_length<Dims>())...);
    }
};

template<class DimensionMap>
struct fix_dm;

template<char ...Dims>
struct fix_dm<dimension_map<index_pair<Dims, std::size_t>...>> {
    static constexpr auto construct(const dimension_map<index_pair<Dims, std::size_t>...> &dm) noexcept {
        return fix<Dims...>(dm.template get<Dims>()...);
    }
};

} // namespace helpers

template<char ...Dims>
struct zip {
private:
    using DimensionMap = helpers::dimension_map<helpers::index_pair<Dims, std::size_t>...>;
    using DimensionMapHelpers = helpers::dimension_map_helpers<Dims...>;

public:
    template<class Struct1, class Struct2>
    constexpr auto operator()(const Struct1 &s1, const Struct2 &s2) const noexcept {
        DimensionMap zero = {};
        DimensionMap lengths1 = DimensionMapHelpers::get_lengths(s1);
        DimensionMap lengths2 = DimensionMapHelpers::get_lengths(s2);
        if(!DimensionMapHelpers::equals(lengths1, lengths2))
            std::terminate();
        bool non_empty = DimensionMapHelpers::all_nonzero(lengths1);
        return helpers::generate(zero, non_empty, [lengths1](DimensionMap& idx){
            return DimensionMapHelpers::increment(idx, lengths1);
        });
    }
};

template<class DimensionMap>
constexpr auto fix(const DimensionMap &idx) noexcept {
    return helpers::fix_dm<DimensionMap>::construct(idx);
}

template<class DimensionMap, class V>
constexpr auto get_at(V *ptr, const DimensionMap &idx) noexcept {
    return compose(fix<DimensionMap>(idx), get_at<V>(ptr));
}

} // namespace noarr

#endif // NOARR_ZIP_HPP
