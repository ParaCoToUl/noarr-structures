#ifndef NOARR_ITERATOR_HPP
#define NOARR_ITERATOR_HPP

#include <tuple>

#include "dimension_map.hpp"
#include "std_ext.hpp"
#include "pipes.hpp"
#include "funcs.hpp"

namespace noarr {

namespace helpers {

template<std::size_t N, char ...Dims>
struct n_th;

template<std::size_t N, char Dim, char ...Dims>
struct n_th<N, Dim, Dims...> : public n_th<N - 1, Dims...> {};

template<char Dim, char ...Dims>
struct n_th<0U, Dim, Dims...> {
    using value_type = char;
    static constexpr value_type value = Dim;
};

template<class Struct, char ...Dims>
struct iterator;

template<std::size_t I, char ...Dims>
struct iterator_eq_helper {
private:
    using DimensionMap = dimension_map<index_pair<Dims, std::size_t>...>;

public:
    static constexpr int compare(const DimensionMap &first, const DimensionMap &second) noexcept {
        constexpr char symbol = n_th<I, Dims...>::value;

        if (first.template get<symbol>() < second.template get<symbol>()) {
            return -1;
        } else if (first.template get<symbol>() > second.template get<symbol>()) {
            return 1;
        } else {
            return iterator_eq_helper<I - 1, Dims...>::compare(first, second);
        }
    }
};

template<char Dim, char ...Dims>
struct iterator_eq_helper<0U, Dim, Dims...> {
private:
    using DimensionMap = dimension_map<index_pair<Dim, std::size_t>, index_pair<Dims, std::size_t>...>;

public:
    static constexpr int compare(const DimensionMap &first, const DimensionMap &second) noexcept {
        if (first.template get<Dim>() < second.template get<Dim>()) {
            return -1;
        } else if (first.template get<Dim>() > second.template get<Dim>()) {
            return 1;
        } else {
            return 0;
        }
    }
};

template<std::size_t I, char ...Dims>
struct iterator_inc_helper {
private:
    using DimensionMap = dimension_map<index_pair<Dims, std::size_t>...>;
    static constexpr char symbol = n_th<I - 1, Dims...>::value;

public:
    template<class Struct>
    static bool increment(DimensionMap &dims, Struct &s) noexcept {
        auto &&dim = dims.template get<symbol>();
        ++dim;

        if (dim == (s | get_length<symbol>())) {
            dim = 0U;

            return iterator_inc_helper<I - 1, Dims...>::increment(dims, s);
        } else {
            return false;
        }
    }
};

template<char ...Dims>
struct iterator_inc_helper<0, Dims...> {
private:
    using DimensionMap = dimension_map<index_pair<Dims, std::size_t>...>;

public:
    template<class Struct>
    static constexpr bool increment(const DimensionMap &, const Struct &) noexcept {
        return true;
    }
};

template<class Struct, char ...Dims>
struct iterator {
    Struct *structure;
    dimension_map<index_pair<Dims, std::size_t>...> dims;

    using value_type = decltype(*structure | fix<Dims...>(const_v<std::size_t>(0U, Dims)...));

private:
    static constexpr std::size_t NDims = sizeof...(Dims);

public:
    constexpr iterator() noexcept : structure(nullptr), dims(const_v<std::size_t>(0U, Dims)...) {}
    constexpr iterator(Struct &structure) noexcept : structure(&structure), dims(const_v<std::size_t>(0U, Dims)...) {}

    template<class ...Idxs>
    constexpr iterator(Struct &structure, Idxs ...idxs) noexcept : structure(&structure), dims(idxs...) {}

    constexpr iterator &operator*() noexcept {
        return *this;
    }

    constexpr const iterator &operator*() const noexcept {
        return *this;
    }

    iterator &operator++() noexcept {
        if (iterator_inc_helper<sizeof...(Dims), Dims...>::increment(dims, *structure)) {
            structure = nullptr;
        }

        return *this;
    }

    constexpr bool operator==(const iterator &other) const noexcept {
        return structure == other.structure && iterator_eq_helper<NDims - 1, Dims...>::compare(dims, other.dims) == 0;
    }

    constexpr bool operator!=(const iterator &other) const noexcept {
        return structure != other.structure || iterator_eq_helper<NDims - 1, Dims...>::compare(dims, other.dims) != 0;
    }
};

template<std::size_t I, class Struct, char ...Dims>
struct iterator_get_helper {
    static constexpr char symbol = n_th<I - 1, Dims...>::value;

    static constexpr decltype(auto) get(noarr::helpers::iterator<Struct, Dims...> &it) noexcept {
        return it.dims.template get<symbol>();
    }

    static constexpr decltype(auto) get(const noarr::helpers::iterator<Struct, Dims...> &it) noexcept {
        return it.dims.template get<symbol>();
    }
};

template<class Struct, char ...Dims>
struct iterator_get_helper<0U, Struct, Dims...> {
    static constexpr decltype(auto) get(noarr::helpers::iterator<Struct, Dims...> &it) noexcept {
        return *it.structure | fix<Dims...>(it.dims.template get<Dims>()...);
    }

    static constexpr decltype(auto) get(const noarr::helpers::iterator<Struct, Dims...> &it) noexcept {
        return *it.structure | fix<Dims...>(it.dims.template get<Dims>()...);
    }
};

template<class Struct, char ...Dims>
struct range {
    Struct structure;
    using Iterator = iterator<Struct, Dims...>;

    range(const Struct &structure) : structure(structure) {}

    Iterator begin() noexcept {
        return Iterator(structure);
    }

    Iterator end() const noexcept {
        return Iterator();
    }
};

} // namespace helpers

template<char ...Dims>
struct iterate {
	using func_family = top_tag;

    template<class Struct>
    constexpr helpers::range<Struct, Dims...> operator()(const Struct &s) const noexcept {
        return helpers::range<Struct, Dims...>(s);
    }
};

} // namespace noarr

namespace std {
    template<class Struct, char ...Dims>
    struct tuple_size<noarr::helpers::iterator<Struct, Dims...>> : integral_constant<std::size_t, sizeof...(Dims) + 1U> {};

    template<class Struct, std::size_t I, char ...Dims>
    struct tuple_element<I, noarr::helpers::iterator<Struct, Dims...>> {
        using type = std::size_t;
    };

    template<class Struct, char ...Dims>
    struct tuple_element<0U, noarr::helpers::iterator<Struct, Dims...>> {
        using type = typename noarr::helpers::iterator<Struct, Dims...>::value_type;
    };

    template<std::size_t I, class Struct, char ...Dims>
    decltype(auto) get(noarr::helpers::iterator<Struct, Dims...> &it) noexcept {
        return noarr::helpers::iterator_get_helper<I, Struct, Dims...>::get(it);
    }
}

#endif // NOARR_ITERATOR_HPP
