#ifndef NOARR_STRUCTURES_TBB_HPP
#define NOARR_STRUCTURES_TBB_HPP

#include <cstdlib>
#include <type_traits>
#include <tbb/tbb.h>

#include "../interop/bag.hpp"
#include "../interop/traverser_iter.hpp"

namespace noarr {

// declared in traverser_iter.hpp
template<IsDim auto Dim, class Struct, class Order>
template<class Split>
constexpr traverser_range_t<Dim, Struct, Order>::traverser_range_t(traverser_range_t &orig, Split) noexcept : base(orig), begin_idx(orig.begin_idx + (orig.end_idx - orig.begin_idx) / 2), end_idx(orig.end_idx) {
	static_assert(std::is_same_v<Split, tbb::split>, "Invalid constructor call");
	orig.end_idx = begin_idx;
}

template<class Traverser, class F>
inline void tbb_for_each(const Traverser &t, const F &f) noexcept {
	tbb::parallel_for(t.range(), [&f](const auto &subrange) { subrange.for_each(f); });
}

template<class Traverser, class FNeut, class FAcc, class FJoin, class OutStruct>
inline void tbb_reduce(const Traverser &t, const FNeut &f_neut, const FAcc &f_acc, const FJoin &f_join, const OutStruct &out_struct, void *out_ptr) noexcept {
	constexpr auto top_dim = helpers::traviter_top_dim<decltype(t.get_struct() ^ t.get_order())>;
	using range_t = decltype(t.range());
	if constexpr(OutStruct::signature::template all_accept<top_dim>) {
		// parallel writes will go to different offsets => out_ptr may be shared
		tbb::parallel_for(t.range(), [&f_acc, out_ptr](const range_t &subrange) {
			subrange.for_each([f_acc, out_ptr](auto state) {
				f_acc(state, out_ptr);
			});
		});
	} else {
		// parallel writes may go to colliding offsets => out_ptr must be privatized
		struct private_ptr {
			void *raw;
			constexpr private_ptr() noexcept : raw(nullptr) {}
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
			if(local_out_ptr == nullptr) {
				local_out_ptr = std::malloc(out_struct.size(empty_state));
				traverser(out_struct).for_each([local_out_ptr, f_neut](auto state) {
					f_neut(state, local_out_ptr);
				});
				local.raw = local_out_ptr;
			}
			subrange.for_each([f_acc, local_out_ptr](auto state) {
				f_acc(state, local_out_ptr);
			});
		});
		out_ptrs.combine_each([out_struct, out_ptr, &f_join](const private_ptr &local_out_ptr) {
			traverser(out_struct).for_each([to=out_ptr, from=local_out_ptr.raw, f_join](auto state) {
				f_join(state, to, (const void *) from);
			});
		});
	}
}

template<class Traverser, class FNeut, class FAcc, class FJoin, class OutBag>
inline void tbb_reduce_bag(const Traverser &t, const FNeut &f_neut, const FAcc &f_acc, const FJoin &f_join, const OutBag &out_bag) noexcept {
	auto out_struct = out_bag.structure();
	return tbb_reduce(t,
		[out_struct, &f_neut](auto out_state, void *out_left) {
			auto bag = make_bag(out_struct, (char *)out_left);
			f_neut(out_state, bag);
		},
		[out_struct, &f_acc](auto in_state, void *out_left) {
			auto bag = make_bag(out_struct, (char *)out_left);
			f_acc(in_state, bag);
		},
		[out_struct, &f_join](auto out_state, void *out_left, const void *out_right) {
			auto left_bag = make_bag(out_struct, (char *)out_left);
			auto right_bag = make_bag(out_struct, (const char *)out_right);
			f_join(out_state, left_bag, right_bag);
		},
		out_struct,
		out_bag.data());
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TBB_HPP
