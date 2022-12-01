#ifndef NOARR_STRUCTURES_TBB_HPP
#define NOARR_STRUCTURES_TBB_HPP

#include <tbb/tbb.h>
#include <tbb/scalable_allocator.h>

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
		struct reducer {
		private:
			const FNeut &f_neut;
			const FEq &f_eq;
			const F &f;
			const OutStruct &out_struct;
			void *const local_out_ptr;
			bool const root;
		public:
			// 1. initialize
			reducer(const FNeut &f_neut, const FEq &f_eq, const F &f, const OutStruct &out_struct, void *out_ptr)
				: f_neut(f_neut), f_eq(f_eq), f(f), out_struct(out_struct), local_out_ptr(out_ptr), root(true) {
			}
			// 2. split
			reducer(const reducer &orig, tbb::split)
				: f_neut(orig.f_neut), f_eq(orig.f_eq), f(orig.f), out_struct(orig.out_struct), local_out_ptr(/*tbb::*/scalable_malloc(orig.out_struct.size(empty_state))), root(false) {
				traverser(out_struct).for_each([ptr=local_out_ptr, f_neut=f_neut](auto state) {
					f_neut(state, ptr);
				});
			}
			// 3. local reduction
			void operator()(const range_t &subrange) {
				subrange.for_each([f_eq=f_eq, local_out_ptr=local_out_ptr](auto... states) {
					f_eq(states..., local_out_ptr);
				});
			}
			// 4. join
			void join(const reducer &other) {
				traverser(out_struct).for_each([to=local_out_ptr, from=other.local_out_ptr, f=f](auto state) {
					f(state, to, (const void *) from);
				});
			}
			// 5. free joined
			~reducer() {
				if(!root)
					/*tbb::*/scalable_free(local_out_ptr);
			}
		} r(f_neut, f_eq, f, out_struct, out_ptr);
		tbb::parallel_reduce(t.range(), r);
	}
}

} // namespace noarr

#endif // NOARR_STRUCTURES_TBB_HPP
