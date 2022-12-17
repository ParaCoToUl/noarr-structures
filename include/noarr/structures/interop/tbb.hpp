#ifndef NOARR_STRUCTURES_TBB_HPP
#define NOARR_STRUCTURES_TBB_HPP

#include <cstdlib>
#include <tbb/tbb.h>

#include "../interop/traverser_iter.hpp"

namespace noarr {

// declared in traverser_iter.hpp
template<char Dim, class Struct, class Order>
template<class Split>
constexpr traverser_range_t<Dim, Struct, Order>::traverser_range_t(traverser_range_t &orig, Split) noexcept : base(orig), begin_idx(orig.begin_idx + (orig.end_idx - orig.begin_idx) / 2), end_idx(orig.end_idx) {
	static_assert(std::is_same_v<Split, tbb::split>, "Invalid constructor call");
	orig.end_idx = begin_idx;
}

template<class Traverser, class F>
inline void tbb_for_each(const Traverser &t, const F &f) noexcept {
	tbb::parallel_for(t.range(), [&f](const auto &subrange) { subrange.for_each(f); });
}

template<class Traverser, class FEq, class F, class OutStruct, class FNeut>
inline void tbb_reduce(const Traverser &t, const FNeut &f_neut, const FEq &f_eq, const F &f, const OutStruct &out_struct, void *out_ptr) noexcept {
	constexpr char top_dim = helpers::traviter_top_dim<decltype(t.get_struct() ^ t.get_order())>;
	using range_t = decltype(t.range());
	if constexpr(OutStruct::signature::template all_accept<top_dim>) {
		// parallel writes will go to different offsets => out_ptr may be shared
		tbb::parallel_for(t.range(), [&f_eq, out_ptr](const range_t &subrange) {
			subrange.for_each([f_eq, out_ptr](auto... states) {
				f_eq(states..., out_ptr);
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
		tbb::parallel_for(t.range(), [&f_neut, &f_eq, &out_struct, &out_ptrs](const range_t &subrange) {
			private_ptr &local = out_ptrs.local();
			void *local_out_ptr = local.raw;
			if(local_out_ptr == nullptr) {
				local_out_ptr = std::malloc(out_struct.size(empty_state));
				traverser(out_struct).for_each([local_out_ptr, f_neut](auto state) {
					f_neut(state, local_out_ptr);
				});
				local.raw = local_out_ptr;
			}
			subrange.for_each([f_eq, local_out_ptr](auto... states) {
				f_eq(states..., local_out_ptr);
			});
		});
		out_ptrs.combine_each([out_struct, out_ptr, &f](const private_ptr &local_out_ptr) {
			traverser(out_struct).for_each([to=out_ptr, from=local_out_ptr.raw, f](auto state) {
				f(state, to, (const void *) from);
			});
		});
	}
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TBB_HPP
