#ifndef NOARR_STRUCTURES_TRAVERSER_ITER_HPP
#define NOARR_STRUCTURES_TRAVERSER_ITER_HPP

#include <cstddef>
#include <iterator>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../extra/traverser.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"

namespace noarr {

template<auto Dim, class Struct, class Order>
requires IsDim<decltype(Dim)>
struct traverser_iterator_t : strict_contain<Struct, Order> {
	using base = strict_contain<Struct, Order>;
	std::size_t idx;

	constexpr traverser_iterator_t(const base &b, std::size_t idx) : base(b), idx(idx) {}

	using this_t = traverser_iterator_t;
	using order_with_fix = decltype(std::declval<Order>() ^ fix<Dim>(std::declval<std::size_t>()));

	[[nodiscard]]
	constexpr auto get_struct() const noexcept {
		return this->template get<0>();
	}

	[[nodiscard]]
	constexpr auto get_order() const noexcept {
		return this->template get<1>();
	}

	using difference_type = std::ptrdiff_t;
	using value_type = traverser_t<Struct, order_with_fix>;
	using reference = value_type;
	using iterator_category = std::random_access_iterator_tag;

	// random_access_iterator must be default constructible, although it does not make sense even for STL iterators.
	// Additionally, it cannot be implemented generally for any Struct.
	// This code should satisfy the trait/concept, but fail at compile-time if actually used in potentially-evaluated
	// context.
private:
	static constexpr const base &construct_base() noexcept {
		static_assert(always_false<this_t>, "Cannot use the default constructor of traverser iterator");
		return construct_base();
	}

public:
	[[deprecated("The default iterator for traversers is not well-definable")]]
	explicit constexpr traverser_iterator_t() noexcept
		: base(construct_base()), idx() {}

	[[nodiscard]]
	constexpr difference_type operator-(const this_t &other) const noexcept {
		return idx - other.idx;
	}

	[[nodiscard]]
	constexpr this_t operator+(difference_type diff) const noexcept {
		return this_t(*this, idx + diff);
	}

	[[nodiscard]]
	constexpr this_t operator-(difference_type diff) const noexcept {
		return this_t(*this, idx - diff);
	}

	constexpr this_t &operator+=(difference_type diff) noexcept {
		idx += diff;
		return *this;
	}

	constexpr this_t &operator-=(difference_type diff) noexcept {
		idx -= diff;
		return *this;
	}

	constexpr this_t &operator++() noexcept {
		idx++;
		return *this;
	}

	constexpr this_t &operator--() noexcept {
		idx--;
		return *this;
	}

	constexpr this_t operator++(int) noexcept {
		const auto copy = *this;
		idx++;
		return copy;
	}

	constexpr this_t operator--(int) noexcept {
		const auto copy = *this;
		idx--;
		return copy;
	}

	[[nodiscard]]
	constexpr bool operator==(const this_t &other) const noexcept {
		return idx == other.idx;
	}

	[[nodiscard]]
	constexpr auto operator<=>(const this_t &other) const noexcept {
		return idx <=> other.idx;
	}

	[[nodiscard]]
	constexpr value_type operator*() const noexcept {
		return value_type(get_struct(), get_order() ^ fix<Dim>(idx));
	}

	[[nodiscard]]
	constexpr value_type operator[](difference_type i) const noexcept {
		return value_type(get_struct(), get_order() ^ fix<Dim>(idx + i));
	}

	[[nodiscard]]
	friend constexpr this_t operator+(difference_type diff, const this_t &iter) noexcept {
		return iter + diff;
	}
};

template<auto Dim, class Struct, class Order>
requires IsDim<decltype(Dim)>
struct traverser_range_t : strict_contain<Struct, Order> {
	using base = strict_contain<Struct, Order>;
	std::size_t begin_idx, end_idx;

	constexpr traverser_range_t(const traverser_t<Struct, Order> &traverser, std::size_t length)
		: base(static_cast<const base &>(traverser)), begin_idx(0), end_idx(length) {}

	// TBB splitting constructor
	template<class Split>
	constexpr traverser_range_t(traverser_range_t &, Split) noexcept; // defined in tbb_traverser.hpp

	[[nodiscard]]
	constexpr auto get_struct() const noexcept {
		return this->template get<0>();
	}

	[[nodiscard]]
	constexpr auto get_order() const noexcept {
		return this->template get<1>();
	}

	template<class NewOrder>
	[[nodiscard]]
	constexpr auto order(NewOrder new_order) const noexcept {
		// equivalent to as_traverser().order(new_order)
		const auto slice_order = slice<Dim>(begin_idx, end_idx - begin_idx);
		return traverser_t<Struct, decltype(get_order() ^ slice_order ^ new_order)>(
			get_struct(), get_order() ^ slice_order ^ new_order);
	}

	template<class F>
	constexpr void for_each(F f) const {
		as_traverser().for_each(f);
	}

	[[nodiscard]]
	constexpr auto as_traverser() const noexcept {
		const auto slice_order = slice<Dim>(begin_idx, end_idx - begin_idx);
		return traverser_t<Struct, decltype(get_order() ^ slice_order)>(get_struct(), get_order() ^ slice_order);
	}

	// empty() and is_divisible() are required by TBB, but it could also be useful to call them directly, so they are
	// implemented here
	[[nodiscard]]
	constexpr bool empty() const noexcept {
		return end_idx == begin_idx;
	}

	[[nodiscard]]
	constexpr bool is_divisible() const noexcept {
		return end_idx - begin_idx > 1;
	}

	using iterator = traverser_iterator_t<Dim, Struct, Order>;
	using const_iterator = iterator;
	using value_type = typename iterator::value_type;
	using reference = typename iterator::reference;
	using const_reference = const reference;
	using difference_type = typename iterator::difference_type;
	using size_type = std::size_t;

	[[nodiscard]]
	constexpr iterator begin() const noexcept {
		return iterator(*this, begin_idx);
	}

	[[nodiscard]]
	constexpr iterator end() const noexcept {
		return iterator(*this, end_idx);
	}

	[[nodiscard]]
	constexpr const_iterator cbegin() const noexcept {
		return const_iterator(*this, begin_idx);
	}

	[[nodiscard]]
	constexpr const_iterator cend() const noexcept {
		return const_iterator(*this, end_idx);
	}

	[[nodiscard]]
	constexpr size_type size() const noexcept {
		return end_idx - begin_idx;
	}

	[[nodiscard]]
	constexpr value_type operator[](size_type i) const noexcept {
		return value_type(get_struct(), get_order() ^ fix<Dim>(begin_idx + i));
	}
};

namespace helpers {

template<class Sig>
struct traviter_sig_top_dim {
	static_assert(
		always_false<Sig>,
		"The top-level dimension (after applying order) must be dynamic in order to be convertible to a range");
};

template<auto Dim, class ArgLength, class RetSig>
requires IsDim<decltype(Dim)>
struct traviter_sig_top_dim<function_sig<Dim, ArgLength, RetSig>> {
	static constexpr auto dim = Dim;
};

template<class Struct>
static constexpr auto traviter_top_dim = traviter_sig_top_dim<typename Struct::signature>::dim;

} // namespace helpers

// declared in traverser.hpp
template<class Struct, class Order>
template<auto Dim>
requires IsDim<decltype(Dim)>
constexpr auto traverser_t<Struct, Order>::range() const noexcept {
	return traverser_range_t<Dim, Struct, Order>(*this, top_struct().template length<Dim>(empty_state));
}

// declared in traverser.hpp
template<class Struct, class Order>
constexpr auto traverser_t<Struct, Order>::range() const noexcept {
	constexpr auto dim = helpers::traviter_top_dim<decltype(top_struct())>;
	return range<dim>();
}

// declared in traverser.hpp
template<class Struct, class Order>
constexpr auto traverser_t<Struct, Order>::begin() const noexcept {
	// same as range().begin()
	constexpr auto dim = helpers::traviter_top_dim<decltype(top_struct())>;
	return traverser_iterator_t<dim, Struct, Order>(*this, 0);
}

// declared in traverser.hpp
template<class Struct, class Order>
constexpr auto traverser_t<Struct, Order>::end() const noexcept {
	// same as range().end()
	constexpr auto dim = helpers::traviter_top_dim<decltype(top_struct())>;
	return traverser_iterator_t<dim, Struct, Order>(*this, top_struct().template length<dim>(empty_state));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TRAVERSER_ITER_HPP
