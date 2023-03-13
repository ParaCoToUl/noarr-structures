#ifndef NOARR_ITERATOR_HPP
#define NOARR_ITERATOR_HPP

#include <tuple>

#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/funcs.hpp"
#include "../structs/setters.hpp"

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

template<char ...Dims>
using iterator_dimension_map = state<state_item<index_in<Dims>, std::size_t>...>;

template<class Struct, char ...Dims>
struct iterator;

template<std::size_t I, char ...Dims>
struct iterator_eq_helper {
private:
	using DimensionMap = iterator_dimension_map<Dims...>;

public:
	static constexpr int compare(const DimensionMap &first, const DimensionMap &second) noexcept {
		using key = index_in<n_th<I, Dims...>::value>;

		if (first.template get<key>() < second.template get<key>()) {
			return -1;
		} else if (first.template get<key>() > second.template get<key>()) {
			return 1;
		} else if constexpr (I == 0) {
			return 0;
		} else {
			return iterator_eq_helper<I - 1, Dims...>::compare(first, second);
		}
	}
};

template<std::size_t I, char ...Dims>
struct iterator_inc_helper {
private:
	using DimensionMap = iterator_dimension_map<Dims...>;
	static constexpr char symbol = n_th<I - 1, Dims...>::value;
	using key = index_in<symbol>;

public:
	template<class Struct>
	static bool increment(DimensionMap &dims, const Struct &s) noexcept {
		auto &dim = dims.template get_ref<key>();
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
	using DimensionMap = iterator_dimension_map<Dims...>;

public:
	template<class Struct>
	static constexpr bool increment(const DimensionMap &, const Struct &) noexcept {
		return true;
	}
};

template<class Struct, char ...Dims>
struct iterator {
	Struct *structure;
	iterator_dimension_map<Dims...> dims;

	using value_type = decltype(*structure ^ fix<Dims...>(((void)Dims, 0U)...));

private:
	static constexpr std::size_t NDims = sizeof...(Dims);

public:
	constexpr iterator() noexcept : structure(nullptr), dims(((void)Dims, 0U)...) {}
	constexpr iterator(Struct *structure) noexcept : structure(structure), dims(((void)Dims, 0U)...) {}

	template<class ...Idxs>
	constexpr iterator(Struct *structure, Idxs ...idxs) noexcept : structure(structure), dims(idxs...) {}

	constexpr auto &operator*() noexcept {
		return dims;
	}

	constexpr auto operator*() const noexcept {
		return dims;
	}

	constexpr auto operator->() noexcept {
		return &dims;
	}

	constexpr auto operator->() const noexcept {
		return &dims;
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

template<class Struct, char ...Dims>
struct range {
	Struct structure;
	using Iterator = iterator<Struct, Dims...>;

	explicit constexpr range(const Struct &structure) noexcept : structure(structure) {}
	explicit constexpr range(Struct &&structure) noexcept : structure(std::move(structure)) {}

	constexpr Iterator begin() noexcept {
		return Iterator(&structure);
	}

	static constexpr Iterator end() noexcept {
		return Iterator();
	}
};

} // namespace helpers

template<char ...Dims>
struct iterate {
	template<class Struct>
	constexpr helpers::range<Struct, Dims...> operator()(const Struct &s) const noexcept {
		return helpers::range<Struct, Dims...>(s);
	}
};

} // namespace noarr

#endif // NOARR_ITERATOR_HPP
