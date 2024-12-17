#ifndef NOARR_STRUCTURES_PLANNER_HPP
#define NOARR_STRUCTURES_PLANNER_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../base/contain.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../structs/views.hpp"
#include "../structs/setters.hpp"
#include "../extra/sig_utils.hpp"
#include "../extra/traverser.hpp"

namespace noarr {

namespace helpers {

template <class Ending>
struct is_activated_impl;

template <class Ending>
using is_activated = is_activated_impl<std::remove_cvref_t<Ending>>;

template <class Ending>
constexpr bool is_activated_v = is_activated<Ending>::value;

} // namespace helpers

template<class Sig, class F>
struct planner_ending_elem_t : flexible_contain<F> {
	using signature = Sig;
	static constexpr bool activated = IsGroundSig<signature>;

	using flexible_contain<F>::flexible_contain;

	template<class NewOrder>
	[[nodiscard]]
	constexpr auto order(NewOrder new_order) const noexcept {
		struct sigholder_t {
			using signature = Sig;
			static_assert(!always_false<signature>);
		};

		return planner_ending_elem_t<typename decltype(sigholder_t() ^ new_order)::signature, F>(flexible_contain<F>::get());
	}

	template<class Planner>
	constexpr void operator()(const Planner &planner) const {
		run(planner, std::make_index_sequence<Planner::num_structs>());
	}

private:
	template<class Planner, std::size_t ...Idxs>
    constexpr void run(const Planner &planner, [[maybe_unused]] std::index_sequence<Idxs...> is) const
	requires (requires(F f) { f(planner.template get_struct<Idxs>()[planner.state()]...); })
	{
		flexible_contain<F>::get()(planner.template get_struct<Idxs>()[planner.state()]...);
	}

	template<class Planner, std::size_t ...Idxs>
    constexpr void run(const Planner &planner, [[maybe_unused]] std::index_sequence<Idxs...> is) const
	requires (requires(F f) { f(planner.state(), planner.template get_struct<Idxs>()[planner.state()]...); })
	{
		flexible_contain<F>::get()(planner.state(), planner.template get_struct<Idxs>()[planner.state()]...);
	}
};

template<class Sig, class F>
struct helpers::is_activated_impl<planner_ending_elem_t<Sig, F>> : std::integral_constant<bool, planner_ending_elem_t<Sig, F>::activated> {};

template<class Sig, class F>
struct planner_ending_t : flexible_contain<F> {
	using signature = Sig;
	static constexpr bool activated = IsGroundSig<signature>;

	using flexible_contain<F>::flexible_contain;

	template<class NewOrder>
	[[nodiscard]]
	constexpr auto order(NewOrder new_order) const noexcept {
		struct sigholder_t {
			using signature = Sig;
			static_assert(!always_false<signature>);
		};

		return planner_ending_t<typename decltype(sigholder_t() ^ new_order)::signature, F>(flexible_contain<F>::get());
	}

	template<class Planner>
	constexpr void operator()(const Planner &planner) const {
		flexible_contain<F>::get()(planner.state());
	}
};

template<class Sig, class F>
struct helpers::is_activated_impl<planner_ending_t<Sig, F>> : std::integral_constant<bool, planner_ending_t<Sig, F>::activated> {};

template<class ...Endings>
struct planner_endings;

template<class ...Endings>
constexpr auto make_planner_endings(Endings &&...endings) noexcept;

template<class ...Endings>
constexpr auto make_planner_endings(flexible_contain<Endings...> endings) noexcept;

template<class ...Endings>
struct planner_endings : flexible_contain<Endings...> {
	static constexpr std::size_t activated_count = (0 + ... + (helpers::is_activated_v<Endings> ? 1 : 0));
	static_assert(activated_count <= 1, "ambiguous activation of planner endings");
	static constexpr bool activated = activated_count == 1;

	using flexible_contain<Endings...>::flexible_contain;

	explicit constexpr planner_endings(flexible_contain<Endings...> contain) noexcept
		: flexible_contain<Endings...>(contain) {}

	template<class NewOrder>
	[[nodiscard]]
	constexpr auto order(NewOrder new_order) const noexcept {
		return order(new_order, std::index_sequence_for<Endings...>());
	}

	template <std::size_t ...Is>
	[[nodiscard]]
	constexpr auto order(auto new_order, [[maybe_unused]] std::integer_sequence<std::size_t, Is...> is) const noexcept {
		return make_planner_endings(this->template get<Is>().order(new_order)...);
	}

	[[nodiscard]]
	static constexpr auto get_next_ending(auto order, auto ending) noexcept {
		if constexpr (helpers::is_activated_v<decltype(ending.order(order))>) {
			return helpers::contain<>();
		} else {
			return helpers::contain(ending);
		}
	}

	[[nodiscard]]
	constexpr auto get_next(auto order) const noexcept requires helpers::is_activated_v<decltype(this->order(order))> {
		return get_next(order, std::index_sequence_for<Endings...>());
	}

	template <std::size_t ...Is>
	[[nodiscard]]
	constexpr auto get_next(auto order, [[maybe_unused]] std::integer_sequence<std::size_t, Is...> is) const noexcept requires helpers::is_activated_v<decltype(this->order(order))> {
		return make_planner_endings(contain_cat(get_next_ending(order, this->template get<Is>())...));
	}

	[[nodiscard]]
	static constexpr std::size_t activated_index() noexcept requires activated {
		return activated_index(std::index_sequence_for<Endings...>());
	}

	template <std::size_t ...Is>
	[[nodiscard]]
	static constexpr std::size_t activated_index([[maybe_unused]] std::integer_sequence<std::size_t, Is...> is) noexcept requires activated {
		return (0 + ... + (helpers::is_activated_v<decltype(std::declval<planner_endings>().template get<Is>())> ? Is : 0));
	}

	[[nodiscard]]
	constexpr auto add_ending(auto ending) const noexcept {
		return add_ending(ending, std::index_sequence_for<Endings...>());
	}

	template <std::size_t ...Is>
	[[nodiscard]]
	constexpr auto add_ending(auto ending, [[maybe_unused]] std::integer_sequence<std::size_t, Is...> is) const noexcept {
		return make_planner_endings(this->template get<Is>()..., ending);
	}

	template<class Planner>
	constexpr void operator()(const Planner &planner) const requires helpers::is_activated_v<decltype(this->order(fix(state_at<typename Planner::union_struct>(planner.top_struct(), empty_state))))> {
		using transformed = decltype(order(fix(state_at<typename Planner::union_struct>(planner.top_struct(), empty_state))));
		this->template get<transformed::activated_index()>()(planner.pop_endings());
	}
};

template<class ...Endings>
struct helpers::is_activated_impl<planner_endings<Endings...>> : std::integral_constant<bool, planner_endings<Endings...>::activated> {};

template<class ...Endings>
planner_endings(Endings &&...) -> planner_endings<std::remove_cvref_t<Endings>...>;

template<class ...Endings>
planner_endings(helpers::contain<Endings...>) -> planner_endings<std::remove_cvref_t<Endings>...>;

template<class ...Endings>
constexpr auto make_planner_endings(Endings &&...endings) noexcept {
	return planner_endings(std::forward<Endings>(endings)...);
}

template<class ...Endings>
constexpr auto make_planner_endings(flexible_contain<Endings...> endings) noexcept {
	return planner_endings(endings);
}

template<class Union, class Order, class Ending>
struct planner_t;

template<class Sig, class F>
struct planner_sections_t : flexible_contain<F> {
	using signature = Sig;
	static constexpr bool activated = IsGroundSig<signature>;

	using flexible_contain<F>::flexible_contain;

	template<class NewOrder>
	[[nodiscard]]
	constexpr auto order(NewOrder new_order) const noexcept {
		struct sigholder_t {
			using signature = Sig;
			static_assert(!always_false<signature>);
		};

		return planner_sections_t<typename decltype(sigholder_t() ^ new_order)::signature, F>(flexible_contain<F>::get());
	}

	template<class Planner>
	constexpr void operator()(const Planner &planner) const {
		flexible_contain<F>::get()(planner);
	}
};

template<class Sig, class F>
struct helpers::is_activated_impl<planner_sections_t<Sig, F>> : std::integral_constant<bool, planner_sections_t<Sig, F>::activated> {};

template<class Union, class Order_, class Ending_>
constexpr auto make_planner(Union &&union_struct, Order_ &&order, Ending_ &&ending) noexcept {
	return planner_t(std::forward<Union>(union_struct), std::forward<Order_>(order), std::forward<Ending_>(ending));
}

template<class ...Structs, class Order, class Ending>
struct planner_t<union_t<Structs...>, Order, Ending> : flexible_contain<union_t<Structs...>, Order, Ending> {
	using union_struct = union_t<Structs...>;
	using order_type = Order;
	using ending_type = Ending;

	using base = flexible_contain<union_struct, Order, Ending>;
	using base::base;

	static constexpr std::size_t num_structs = sizeof...(Structs);

	[[nodiscard]]
	constexpr union_struct get_union() const noexcept { return base::template get<0>(); }

	[[nodiscard]]
	constexpr Order get_order() const noexcept { return base::template get<1>(); }

	[[nodiscard]]
	constexpr Ending get_ending() const noexcept { return base::template get<2>(); }

	template <std::size_t Idx> requires (Idx < num_structs)
	[[nodiscard]]
	constexpr auto get_struct() const noexcept { return get_union().template get<Idx>(); }

	template<class NewOrder>
	[[nodiscard("returns a new planner")]]
	constexpr auto order(NewOrder new_order) const noexcept {
		return make_planner(get_union(), get_order() ^ new_order, get_ending());
	}

	[[nodiscard("returns a new planner")]]
	constexpr auto pop_endings() const noexcept {
		return make_planner(get_union(), get_order(), get_ending().get_next(fix(state_at<union_struct>(top_struct(), empty_state))));
	}

	template<class F> requires (std::same_as<Ending, planner_endings<>>)
	[[nodiscard("returns a new planner")]]
	constexpr auto for_each(F f) const {
		using signature = typename decltype(get_union())::signature;
		return make_planner(get_union(), get_order(), planner_endings(planner_ending_t<signature, F>(f)));
	}

	template<class F> requires (std::same_as<Ending, planner_endings<>>)
	[[nodiscard("returns a new planner")]]
	constexpr auto for_each_elem(F f) const {
		using signature = typename decltype(get_union())::signature;
		return make_planner(get_union(), get_order(), planner_endings(planner_ending_elem_t<signature, F>(f)));
	}

	template<auto ...Dims, class F> requires IsDimPack<decltype(Dims)...>
	[[nodiscard("returns a new planner")]]
	constexpr auto for_sections(F f) const {
		using union_sig = typename decltype(get_union())::signature;
		struct sigholder_t {
			using signature = union_sig;
			static_assert(!always_false<signature>); // suppresses warnings
		};

		using signature = typename decltype(sigholder_t() ^ reorder<Dims...>())::signature;

		return make_planner(get_union(), get_order(), get_ending().add_ending(planner_sections_t<signature, F>(f)));
	}

	constexpr void execute() const requires (!std::same_as<Ending, planner_endings<>>) {
		using dim_tree = sig_dim_tree<typename decltype(top_struct())::signature>;
		for_each_impl(dim_tree(), empty_state);
	}

	// execute the planner
	constexpr void operator()() const requires (!std::same_as<Ending, planner_endings<>>) {
		execute();
	}

	[[nodiscard("returns the state of the planner")]]
	constexpr auto state() const noexcept {
		return state_at<union_struct>(top_struct(), empty_state);
	}

	[[nodiscard("returns the top struct of the planner")]]
	constexpr auto top_struct() const noexcept {
		return get_union() ^ get_order();
	}

private:
	template<auto Dim, class Branch, class ...Branches, std::size_t I, std::size_t ...Is>
	constexpr void for_each_impl_dep(auto state, [[maybe_unused]] std::index_sequence<I, Is...> is) const {
		for_each_impl(Branch(), state.template with<index_in<Dim>>(std::integral_constant<std::size_t, I>()));
		for_each_impl_dep<Dim, Branches...>(state, std::index_sequence<Is...>());
	}
	template<auto Dim, class F>
	constexpr void for_each_impl_dep([[maybe_unused]] F f, [[maybe_unused]] auto state, [[maybe_unused]] std::index_sequence<> is) const {}

	template<auto Dim, class ...Branches, IsState State>
	constexpr void for_each_impl([[maybe_unused]] dim_tree<Dim, Branches...> dt, State state) const {
		if constexpr (helpers::is_activated_v<decltype(get_ending().order(fix(state_at<union_struct>(top_struct(), state))))>) {
			get_ending()(order(fix(state)));
		} else {
			using dim_sig = sig_find_dim<Dim, State, typename decltype(top_struct())::signature>;
			if constexpr(dim_sig::dependent) {
				for_each_impl_dep<Dim, Branches...>(state, std::index_sequence_for<Branches...>());
			} else {
				std::size_t len = top_struct().template length<Dim>(state);
				for(std::size_t i = 0; i < len; i++) {
					for_each_impl(Branches()..., state.template with<index_in<Dim>>(i));
				}
			}
		}
	}
	template<IsState State>
	constexpr void for_each_impl([[maybe_unused]] dim_sequence<> ds, State state) const {
		static_assert(helpers::is_activated_v<decltype(get_ending().order(fix(state_at<union_struct>(top_struct(), state))))>);
		get_ending()(order(fix(state)));
	}
};

template<class Union, class Order, class Ending>
planner_t(Union &&, Order &&, Ending &&) -> planner_t<std::remove_cvref_t<Union>, std::remove_cvref_t<Order>, std::remove_cvref_t<Ending>>;

template<class ...Ts>
[[nodiscard("returns a new planner")]]
constexpr auto planner(const Ts &...s) noexcept
{ return planner(make_union(s.get_ref()...)); }

template<class T>
struct is_planner : std::false_type {};

template<class Union, class Order, class Ending>
struct is_planner<planner_t<Union, Order, Ending>> : std::true_type {};

template<class T>
struct is_planner<const T> : is_planner<T> {};

template<class T>
constexpr bool is_planner_v = is_planner<T>::value;

template<class T>
concept IsPlanner = is_planner_v<std::remove_cvref_t<T>>;

template<class ...Ts>
constexpr planner_t<union_t<Ts...>, neutral_proto, planner_endings<>> planner(union_t<Ts...> u) noexcept { return planner_t<union_t<Ts...>, neutral_proto, planner_endings<>>(u, neutral_proto(), planner_endings<>()); }

template<IsPlanner P>
constexpr auto operator^(const P &p, IsProtoStruct auto order) noexcept {
	return p.order(order);
}

template<IsPlanner P>
struct to_state<P> {
	using type = std::remove_cvref_t<decltype(std::declval<P>().state())>;

	[[nodiscard]]
	static constexpr type convert(const P &p) noexcept { return p.state(); }
};

namespace helpers {

template<class F, auto ...Dims> requires (... && IsDim<decltype(Dims)>)
struct for_each_elem_t : public F {};


} // namespace helpers

struct planner_execute_t {};

constexpr planner_execute_t planner_execute() noexcept { return {}; }

template<auto ...Dims, class F> requires (... && IsDim<decltype(Dims)>)
constexpr auto for_each_elem(F &&f) noexcept {
	return helpers::for_each_elem_t<std::remove_cvref_t<F>, Dims...>{std::forward<F>(f)};
}

template<IsPlanner P, class F>
constexpr auto operator^(const P &p, const helpers::for_each_t<F> &f) -> decltype(p.for_each(f)) {
	return p.for_each(f);
}

template<IsPlanner P, class F>
constexpr auto operator^(const P &p, const helpers::for_each_elem_t<F> &f) -> decltype(p.for_each_elem(f)) {
	return p.for_each_elem(f);
}

template<IsPlanner P, auto ...Dims, class F>
constexpr auto operator^(const P &p, const helpers::for_sections_t<F, Dims...> &f) -> decltype(p.template for_sections<Dims...>(f)) {
	return p.template for_sections<Dims...>(f);
}

template<IsPlanner P, auto ...Dims, class F> requires (sizeof...(Dims) > 0)
constexpr auto operator^(const P &p, const helpers::for_dims_t<F, Dims...> &f) -> decltype(p.template for_sections<Dims...>(f).order(hoist<Dims...>())) {
	return p.template for_sections<Dims...>(f).order(hoist<Dims...>());
}

template<IsPlanner P, class F>
constexpr auto operator^(const P &p, const helpers::for_dims_t<F> &f) -> decltype(p.for_sections(f)) {
	return p.for_sections(f);
}

template<IsPlanner P>
constexpr void operator|(const P &p, [[maybe_unused]] planner_execute_t exec) {
	p.execute();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_PLANNER_HPP
