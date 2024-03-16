#ifndef NOARR_STRUCTURES_INTEROP_PLANNER_ITER_HPP
#define NOARR_STRUCTURES_INTEROP_PLANNER_ITER_HPP

#include <cstddef>
#include <iterator>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../extra/planner.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"
#include "noarr/structures/base/utility.hpp"
#include "noarr/structures/interop/traverser_iter.hpp"

namespace noarr {

template<IsDim auto Dim, class Union, class Order, class Ending>
struct planner_iterator_t;

template<IsDim auto Dim, class ...Structs, class Order, class Ending>
struct planner_iterator_t<Dim, union_t<Structs...>, Order, Ending> : strict_contain<union_t<Structs...>, Order, Ending> {
	using this_t = planner_iterator_t;
	using base = strict_contain<union_t<Structs...>, Order, Ending>;
	std::size_t idx;

	using union_struct = union_t<Structs...>;
	using order_type = Order;
	using ending_type = Ending;

	using order_with_fix = decltype(std::declval<Order>() ^ fix<Dim>(std::declval<std::size_t>()));

	static constexpr std::size_t num_structs = sizeof...(Structs);

	[[nodiscard]]
	constexpr auto get_union() const noexcept { return this->template get<0>(); }
	[[nodiscard]]
	constexpr auto get_order() const noexcept { return this->template get<1>(); }
	[[nodiscard]]
	constexpr auto get_ending() const noexcept { return this->template get<2>(); }

	template<std::size_t I>
	[[nodiscard]]
	constexpr auto get_struct() const noexcept { return get_union().template get<I>(); }

	using difference_type = std::ptrdiff_t;
	using value_type = planner_t<union_t<Structs...>, order_with_fix, Ending>;
	using reference = value_type;
	using iterator_category = std::random_access_iterator_tag;

	constexpr planner_iterator_t(const base &b, std::size_t idx) : base(b), idx(idx) {}

	// random_access_iterator must be default constructible, although it does not make sense even for STL iterators.
	// Additionally, it cannot be implemented generally for any Struct.
	// This code should satisfy the trait/concept, but fail at compile-time if actually used in potentially-evaluated context.
private:
	static constexpr const base &construct_base() noexcept {
		static_assert(always_false<this_t>, "Cannot use the default constructor of planner iterator");
		return construct_base();
	}

public:
	[[deprecated("The default iterator for planners is not well-definable")]] explicit constexpr planner_iterator_t() noexcept : base(construct_base()), idx() {}

	constexpr difference_type operator-(const this_t &other) const noexcept { return idx - other.idx; }
	constexpr this_t operator+(difference_type diff) const noexcept { return this_t(*this, idx + diff); }
	constexpr this_t operator-(difference_type diff) const noexcept { return this_t(*this, idx - diff); }
	constexpr this_t &operator+=(difference_type diff) noexcept { idx += diff; return *this; }
	constexpr this_t &operator-=(difference_type diff) noexcept { idx -= diff; return *this; }
	constexpr this_t &operator++() noexcept { idx++; return *this; }
	constexpr this_t &operator--() noexcept { idx--; return *this; }
	this_t operator++(int) noexcept { const auto copy = *this; idx++; return copy; }
	this_t operator--(int) noexcept { const auto copy = *this; idx--; return copy; }

	constexpr bool operator==(const this_t &other) const noexcept { return idx == other.idx; }
	constexpr auto operator<=>(const this_t &other) const noexcept { return idx <=> other.idx; }

	constexpr value_type operator*() const noexcept { return value_type(get_union(), get_order() ^ fix<Dim>(idx), get_ending()); }
	constexpr value_type operator[](difference_type i) const noexcept { return value_type(get_union(), get_order() ^ fix<Dim>(idx + i), get_ending()); }

	friend constexpr this_t operator+(difference_type diff, const this_t &iter) noexcept { return iter + diff; }
};

template<class ...Structs, class Order, class Ending>
constexpr auto begin(const planner_t<union_t<Structs...>, Order, Ending> &planner) {
	constexpr auto dim = helpers::traviter_top_dim<decltype(planner.top_struct())>;
	return planner_iterator_t<dim, union_t<Structs...>, Order, Ending>(planner, 0);
}

template<class ...Structs, class Order, class Ending>
constexpr auto end(const planner_t<union_t<Structs...>, Order, Ending> &planner) {
	constexpr auto dim = helpers::traviter_top_dim<decltype(planner.top_struct())>;
	return planner_iterator_t<dim, union_t<Structs...>, Order, Ending>(planner, planner.top_struct().template length<dim>(empty_state));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_PLANNER_ITER_HPP
