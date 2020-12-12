#include <cassert>
#include <cstddef>

namespace arrr {

/* recursion helper */
template <char... CS> constexpr bool dims_empty() { return false; }
template <> constexpr bool dims_empty<>() { return true; }

/* this has size 0, empty structs have otherwise size 1 */
struct empty_struct_t {
  int empty_struct_dummy[0];
};

/* tool for finding uniquely named indices in index lists */
template <char C, char... CS> struct idx_find {
  empty_struct_t empty_struct;

  template <char DIM, typename... Args>
  static constexpr size_t find(size_t idx, Args... idxs) {
    if constexpr (DIM == C) {
      static_assert(!idx_find<CS...>::template contains<DIM>(),
                    "redundant index");
      return idx;
    } else
      return idx_find<CS...>::template find<DIM>(idxs...);
  }

  template <char DIM> static constexpr bool contains() {
    return DIM == C || idx_find<CS...>::template contains<DIM>();
  }
};

template <char C> struct idx_find<C> {
  empty_struct_t empty_struct;

  template <char DIM> static constexpr size_t find(size_t idx) {
    static_assert(DIM == C, "missing index");
    return idx;
  }

  template <char DIM> static constexpr bool contains() { return DIM == C; }
};

/* basic type kinds */
template <typename T> struct scalar {
  empty_struct_t empty_struct;

  constexpr size_t step() const { return sizeof(T); }
  constexpr size_t size() const { return sizeof(T); }

  template <char... CS> static constexpr bool has_idxs() { return false; }
  template <char C> static constexpr bool has_idx() { return false; }

  template <char C> scalar<T> resize(size_t) const { return scalar<T>(); }

  constexpr size_t idx() const { return idx_(); }

  template <char... CS, typename... Args>
  constexpr size_t idx_(Args... a) const {
    return 0;
  }
};

template <char DIM, typename T, typename IMPL> struct dimension {
  T t;
  IMPL impl;

  dimension() {}

  template <typename... Args> dimension(T t, Args... a) : t(t), impl(a...) {}

  constexpr size_t step() const { return impl.step(t); }
  constexpr size_t size() const { return impl.size(t); }

  template <char C, char... CS> static constexpr bool has_idxs() {
    if constexpr (dims_empty<CS...>())
      return has_idx<C>();
    else
      return has_idx<C>() && has_idxs<CS...>();
  }

  template <char C> static constexpr bool has_idx() {
    if constexpr (DIM == C) {
      static_assert(!T::template has_idx<C>(), "redundant index");
      return true;
    }
    return T::template has_idx<C>();
  }

  template <char C> auto resize(size_t n) const -> auto {
    if constexpr (DIM == C)
      return dimension<DIM, T, decltype(impl.resize(0))>(t, impl.resize(n));
    else
      return dimension<DIM, decltype(t.template resize<C>(0)), IMPL>(
          t.template resize<C>(n), impl);
  }

  template <char... CS, typename... Args>
  constexpr size_t idx(Args... a) const {
    static_assert(has_idxs<CS...>(), "unknown index");
    return idx_<CS...>(a...);
  }

  template <char... CS, typename... Args>
  constexpr size_t idx_(Args... a) const {
    return idx_find<CS...>::template find<DIM>(a...) * step() +
           t.template idx_<CS...>(a...);
  }
};

/* TODO (far future) extra kinds:
 * - dimension2 (eats 2 dims at once (is that even required with transform21?))
 * - tuple (chooses a conditional branch based on dim presence; likely needs a
 * slight expansion of has_idx mechanism)
 * - transform12 (eats a dim, creates 2, e.g. for Z-order expansion or for block
 * splits)
 * - transform21 (eats 2 dims, creates 1, e.g. for Z-order contraction or
 * triangles/symmetries)
 */

/* container implementations */

template <size_t N> struct array_impl {
  empty_struct_t empty_struct;

  array_impl resize(size_t) const { return array_impl<N>(); }
  template <typename T> constexpr size_t step(T t) const { return t.size(); }
  template <typename T> constexpr size_t size(T t) const { return N * step(t); }
  template <typename T> constexpr size_t index(size_t i) const { return i; }
};

struct vector_impl {
  size_t n;
  vector_impl(size_t n) : n(n) {}
  vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T> constexpr size_t step(T t) const { return t.step(); }
  template <typename T> constexpr size_t size(T t) const { return n * step(t); }
  template <typename T> constexpr size_t index(size_t i) const { return i; }
};

struct vector_impl_unsized {
  empty_struct_t empty_struct;

  vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T> constexpr size_t step(T t) const { return t.step(); }
  template <typename T> constexpr size_t index(size_t i) const { return i; }
};

/* TODO (far future) extra containers:
 * - zorder, hilbert (with transform21&12)
 * - triangle (transform21)
 */

/* user-facing container shortcuts */

template <char DIM, size_t N, typename T>
using array = dimension<DIM, T, array_impl<N>>;

template <char DIM, typename T>
using vector = dimension<DIM, T, vector_impl_unsized>;

}; // namespace arrr

/* TESTS */
#include <iostream>
#include <typeinfo>
using namespace arrr;
using std::cout;
using std::endl;

int main() {
  cout << "finder tests:" << endl;
  cout << idx_find<'c', 'd'>::contains<'c'>() << endl;
  cout << idx_find<'c', 'd'>::contains<'d'>() << endl;
  cout << idx_find<'c', 'd'>::contains<'e'>() << endl;
  cout << idx_find<'c', 'd'>::find<'c'>(1, 2) << endl;
  cout << idx_find<'c', 'd'>::find<'d'>(1, 2) << endl;
  // fails:
  // std::cout << idx_find<'c', 'd'>::find<'e'>(1, 2) << std::endl;

  array<'x', 20, vector<'y', scalar<char>>> a;
  cout << "array tests:" << endl;
  cout << a.has_idx<'x'>() << endl;
  cout << a.has_idx<'y'>() << endl;
  cout << a.has_idx<'z'>() << endl;
  cout << a.has_idxs<'x'>() << endl;
  cout << a.has_idxs<'x', 'y'>() << endl;
  cout << a.has_idxs<'x', 'y', 'z'>() << endl;
  // fails b/c unsized vector
  // cout << a.idx<'x','y'>(6,4) << endl;

  auto as = a.resize<'y'>(30);

  cout << as.idx<'y', 'x'>(6, 5) << endl;
  // fails:
  // cout << as.idx<'y'>(6,5) << endl;
  // cout << as.idx<'y'>(6) << endl;
  // cout << as.idx<'y','x'>(6,5,4) << endl;
  // cout << as.idx<'z','y','x'>(6,5,4) << endl;

  cout << "sizes:" << endl;
  cout << sizeof(a) << endl;
  cout << sizeof(as) << endl;
  cout << typeid(a).name() << endl;
  cout << typeid(as).name() << endl;
  return 0;
}
