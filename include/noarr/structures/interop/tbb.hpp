#ifndef NOARR_STRUCTURES_TBB_HPP
#define NOARR_STRUCTURES_TBB_HPP

#include <cstdlib>

#include <type_traits>

#include <tbb/tbb.h>

#include "../base/state.hpp"
#include "../base/utility.hpp"
#include "../extra/planner.hpp"
#include "../extra/traverser.hpp"
#include "../interop/bag.hpp"
#include "../interop/planner_iter.hpp"
#include "../interop/traverser_iter.hpp"

namespace noarr {

// declared in traverser_iter.hpp
template<auto Dim, class Struct, class Order>
requires IsDim<decltype(Dim)>
template<class Split>
constexpr traverser_range_t<Dim, Struct, Order>::traverser_range_t(traverser_range_t &orig, Split /*unused*/) noexcept
	: base(orig), begin_idx(orig.begin_idx + (orig.end_idx - orig.begin_idx) / 2), end_idx(orig.end_idx) {
	static_assert(std::is_same_v<Split, tbb::split>, "Invalid constructor call");
	orig.end_idx = begin_idx;
}

template<IsTraverser Traverser, class F>
inline void tbb_for_each(const Traverser &t, const F &f) {
	tbb::parallel_for(t.range(), [&f](const auto &subrange) { subrange.for_each(f); });
}

template<IsTraverser Traverser, class F>
inline void tbb_for_sections(const Traverser &t, const F &f) {
	tbb::parallel_for(t.range(), [&f](const auto &subrange) {
		constexpr auto top_dim = helpers::traviter_top_dim<decltype(subrange.get_struct() ^ subrange.get_order())>;
		subrange.as_traverser().template for_sections<top_dim>(f);
	});
}

template<IsTraverser Traverser, class FNeut, class FAcc, class FJoin, class OutStruct>
inline void tbb_reduce(const Traverser &t, const FNeut &f_neut, const FAcc &f_acc, const FJoin &f_join,
                       const OutStruct &out_struct, void *out_ptr) {
	constexpr auto top_dim = helpers::traviter_top_dim<decltype(t.get_struct() ^ t.get_order())>;
	using range_t = decltype(t.range());
	if constexpr (OutStruct::signature::template all_accept<top_dim>) {
		// parallel writes will go to different offsets => out_ptr may be shared
		tbb::parallel_for(t.range(), [&f_acc, out_ptr](const range_t &subrange) {
			subrange.for_each([f_acc, out_ptr](auto state) { f_acc(state, out_ptr); });
		});
	} else {
		// parallel writes may go to colliding offsets => out_ptr must be privatized
		struct private_ptr {
			void *raw = nullptr;

			constexpr private_ptr() noexcept = default;
			private_ptr(const private_ptr &) = delete;
			private_ptr(private_ptr &&) = delete;
			private_ptr &operator=(const private_ptr &) = delete;
			private_ptr &operator=(private_ptr &&) = delete;

			~private_ptr() { std::free(raw); } // ok with null
		};

		tbb::combinable<private_ptr> out_ptrs;
		tbb::parallel_for(t.range(), [&f_neut, &f_acc, &out_struct, &out_ptrs](const range_t &subrange) {
			private_ptr &local = out_ptrs.local();
			void *local_out_ptr = local.raw;
			if (local_out_ptr == nullptr) {
				local_out_ptr = std::malloc(out_struct.size(empty_state));
				traverser(out_struct).for_each([local_out_ptr, f_neut](auto state) { f_neut(state, local_out_ptr); });
				local.raw = local_out_ptr;
			}
			subrange.for_each([f_acc, local_out_ptr](auto state) { f_acc(state, local_out_ptr); });
		});
		out_ptrs.combine_each([out_struct, out_ptr, &f_join](const private_ptr &local_out_ptr) {
			traverser(out_struct).for_each([to = out_ptr, from = local_out_ptr.raw, f_join](auto state) {
				f_join(state, to, (const void *)from);
			});
		});
	}
}

template<IsTraverser Traverser, class FNeut, class FAcc, class FJoin, IsBag OutBag>
inline void tbb_reduce(const Traverser &t, const FNeut &f_neut, const FAcc &f_acc, const FJoin &f_join,
                       const OutBag &out_bag) {
	const auto out_struct = out_bag.structure();
	return tbb_reduce(
		t,
		[out_struct, &f_neut](auto out_state, void *out_left) {
			const auto bag = make_bag(out_struct, out_left);
			f_neut(out_state, bag);
		},
		[out_struct, &f_acc](auto in_state, void *out_left) {
			const auto bag = make_bag(out_struct, out_left);
			f_acc(in_state, bag);
		},
		[out_struct, &f_join](auto out_state, void *out_left, const void *out_right) {
			const auto left_bag = make_bag(out_struct, out_left);
			const auto right_bag = make_bag(out_struct, out_right);
			f_join(out_state, left_bag, right_bag);
		},
		out_struct, out_bag.data());
}

struct planner_tbb_execute_t {};

constexpr planner_tbb_execute_t planner_tbb_execute() noexcept { return {}; }

template<IsPlanner Planner>
inline void operator|(const Planner &planner, planner_tbb_execute_t /*unused*/) {
	tbb::parallel_for(range(planner), [](const auto &sub_range) { sub_range.as_planner().execute(); });
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TBB_HPP
