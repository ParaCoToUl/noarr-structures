#include <cassert>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <tuple>
#include <utility>

namespace arrr {

/* this has size 0, "plain empty" structs that do not include this have size 1
 */
struct empty_struct_t {
  int empty_struct_dummy[0] = {};
};

/* polymorphic dot-style function application, avoids lisp))))))) */
template <typename F, typename K>
constexpr auto operator%(K k, F f) {
  return f(k);
}

/*
 * Basic type kinds!
 */

/* scalars */
template <typename T>
struct scalar_t {
  empty_struct_t empty_struct;

  using scalar_type = T;

  constexpr scalar_t() {}
};

/* 1-dimensions with char name */
template <char DIM, typename T, typename IMPL>
struct dimension {
  T t;
  IMPL impl;

  using scalar_type = typename T::scalar_type;

  constexpr dimension() : t(), impl() {}

  template <typename... Args>
  constexpr dimension(T t, Args... a) : t(t), impl(a...) {}
};

/* dimension index fix */
template <char DIM, typename T>
struct fixed_dimension {
  size_t idx;
  T t;

  using scalar_type = typename T::scalar_type;

  constexpr fixed_dimension() : idx(), t() {}

  template <typename... Args>
  constexpr fixed_dimension(size_t idx, Args... a) : idx(idx), t(a...) {}
};

/* converts a single dimension to 2 */
template <char A, char X, char Y, typename T, typename IMPL>
struct split2 {
  T t;
  IMPL impl;

  using scalar_type = typename T::scalar_type;

  constexpr split2() {}
  template <typename... Args>
  constexpr split2(T t, Args... a) : t(t), impl(a...) {}
};

/* merges 2 dimensions into 1 */
template <char A, char B, char X, typename T, typename IMPL>
struct join2 {
  T t;
  IMPL impl;

  using scalar_type = typename T::scalar_type;

  constexpr join2() {}
  template <typename... Args>
  constexpr join2(T t, Args... a) : t(t), impl(a...) {}
};

/* tuple machinery */
template <char DIM, size_t I, typename... TS>
struct tuple_dim {
  using TUP = std::tuple<TS...>;
  TUP ts;

  template <size_t J>
  using tuple_element = typename std::tuple_element<J, TUP>::type;

  using current_element = tuple_element<I>;
  using scalar_type = typename tuple_element<I>::scalar_type;

  constexpr tuple_dim() {}
  constexpr tuple_dim(TS... ts) : ts(ts...) {}
  constexpr tuple_dim(TUP tup) : ts(tup) {}
};

/* tuple type modifications; thanks @sebrockm on stackoverflow
 * TODO: we can likely remove some of the casts */
template <size_t B, size_t E, typename T>
constexpr auto tuple_slice(T t) {
  return [&]<size_t... IS>(std::index_sequence<IS...>) {
    return std::make_tuple(std::get<B + IS>(t)...);
  }
  (std::make_index_sequence<E - B>());
}

template <std::size_t I, typename T, typename NT>
constexpr auto tuple_set_element(T tuple, NT t) {
  constexpr auto N = std::tuple_size_v<std::remove_reference_t<T>>;
  return std::tuple_cat(tuple_slice<0, I>(tuple), std::make_tuple(t),
                        tuple_slice<I + 1, N>(tuple));
}

/*
 * Functions on type kinds!
 */

struct size_f {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr size_t operator()(scalar_t<T> s) {
    return sizeof(T);
  }

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return d.impl.size(d.t);
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t % size_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t % size_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t % size_f();
  }

  static constexpr size_t tuple_size1() { return 0; }

  template <typename T, typename... TS>
  static constexpr size_t tuple_size1(T t, TS... ts) {
    return t % size_f() + tuple_size1(ts...);
  }

  static constexpr size_t tuple_size(std::tuple<> t) { return tuple_size1(); }

  template <typename... TS>
  static constexpr size_t tuple_size(std::tuple<TS...> t) {
    return std::apply(tuple_size1<TS...>, t);
  }

  template <char DIM, size_t I, typename... TS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t) {
    return tuple_size(t.ts);
  }
};

static constexpr size_f size;

/* demo: total layers of construction all the way to a scalar */
struct depth_f {
  empty_struct_t empty_struct;
  template <typename T>
  constexpr size_t operator()(scalar_t<T> s) {
    return 0;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return 1 + d.t % depth_f();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t % depth_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t % depth_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t % depth_f();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t) {
    return 1 + std::get<I>(t.ts) % depth_f();
  }
};

static constexpr depth_f depth;

/* reverse depth -- construction layers between the index and a base scalar */
template <char C>
struct level_f {
  empty_struct_t empty_struct;

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return d.t % depth_f();
    else
      return d.t % level_f<C>();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t % level_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    if constexpr (A == C)
      return s.t % depth_f();
    else
      return s.t % level_f<C>();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    if constexpr (A == C || B == C)
      return j.t % depth_f();
    else
      return j.t % level_f<C>();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t) {
    if constexpr (DIM == C)
      return std::get<I>(t.ts) % depth_f();
    else
      return std::get<I>(t.ts) % level_f<C>();
  }
};

template <char C>
static constexpr level_f<C> level;

/* completely typelevel-predicate tool for checking dimension reachability
 * (TODO: several other functions can be typelevel too) */
template <char C, typename T>
struct has_dim_p;

template <char C, typename T>
struct has_dim_p<C, scalar_t<T>> {
  static constexpr bool get() { return false; }
};

template <char C, char DIM, typename T, typename IMPL>
struct has_dim_p<C, dimension<DIM, T, IMPL>> {
  static constexpr bool get() {
    if constexpr (C == DIM) {
      static_assert(!has_dim_p<C, T>::get(), "redundant dimension");
      return true;
    } else
      return has_dim_p<C, T>::get();
  }
};

template <char C, char DIM, typename T>
struct has_dim_p<C, fixed_dimension<DIM, T>> {
  static constexpr bool get() { return has_dim_p<C, T>::get(); }
};

template <char C, char A, char X, char Y, typename T, typename IMPL>
struct has_dim_p<C, split2<A, X, Y, T, IMPL>> {
  static constexpr bool get() {
    return C == A || (C != X && C != Y && has_dim_p<C, T>::get());
  }
};

template <char C, char A, char B, char X, typename T, typename IMPL>
struct has_dim_p<C, join2<A, B, X, T, IMPL>> {
  static constexpr bool get() {
    return C == A || C == B || (C != X && has_dim_p<C, T>::get());
  }
};

template <char C, char DIM, size_t I, typename... TS>
struct has_dim_p<C, tuple_dim<DIM, I, TS...>> {
  using TI = typename tuple_dim<DIM, I, TS...>::current_element;
  static constexpr bool get() {
    return has_dim_p<C, TI>::get();  // TRICKY: tuple "dimensions" are distinct
                                     // from normal dimensions
  }
};

/* value interface for the above */
template <char C>
struct has_dim_f {
  empty_struct_t empty_struct;

  template <typename K>
  constexpr bool operator()(K) {
    return has_dim_p<C, K>::get();
  }
};

template <char C>
static constexpr has_dim_f<C> has_dim;

/* does the structure have all of given character indexes? */
template <char C, char... CS>
struct has_dims_f {
  empty_struct_t empty_struct;

  template <typename K>
  constexpr bool operator()(K k) {
    return k % has_dim_f<C>() && k % has_dims_f<CS...>();
  }
};

template <char C>
struct has_dims_f<C> {
  empty_struct_t empty_struct;

  template <typename K>
  constexpr bool operator()(K k) {
    return k % has_dim_f<C>();
  }
};

template <char... CS>
static constexpr has_dims_f<CS...> has_dims;

/* variadic args & std::tuple to tuple_dim conversion helper */
template <char DIM, size_t I>
struct make_tuple_dim_f {
  template <typename... TS>
  constexpr tuple_dim<DIM, I, TS...> operator()(TS... ts) const {
    return tuple_dim<DIM, I, TS...>(ts...);
  }
};

template <char DIM, size_t I>
static constexpr make_tuple_dim_f<DIM, I> make_tuple_dim;

template <char DIM, size_t I>
struct make_tuple_f {
  template <typename... TS>
  constexpr tuple_dim<DIM, I, TS...> operator()(std::tuple<TS...> ts) const {
    return tuple_dim<DIM, I, TS...>(ts);
  }
};

template <char DIM, size_t I>
static constexpr make_tuple_f<DIM, I> make_tuple;

/* resize unsized_vectors to sized vectors
 * TODO: unsize, unsize_all */
template <char C>
struct resize {
  empty_struct_t empty_struct;
  size_t n;

  constexpr resize(size_t n) : n(n) {}

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return dimension<DIM, T, decltype(d.impl.resize(n))>(d.t,
                                                           d.impl.resize(n));
    else
      return dimension<DIM, decltype(d.t % resize<C>(n)), IMPL>(
          d.t % resize<C>(n), d.impl);
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return d.t % resize<C>(n);  // FIXME this removes the fix, is that right?
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    if constexpr (A == C)
      return split2<A, X, Y, T, decltype(s.impl.resize(n))>(s.t,
                                                            s.impl.resize(n));
    else
      return split2<A, X, Y, decltype(s.t % resize<C>(n)), IMPL>(
          s.t % resize<C>(n), s.impl);
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    if constexpr (A == C)
      return join2<A, B, X, T, decltype(j.impl.resize1(n))>(j.t,
                                                            j.impl.resize1(n));
    else if constexpr (B == C)
      return join2<A, B, X, T, decltype(j.impl.resize2(n))>(j.t,
                                                            j.impl.resize2(n));
    else
      return join2<A, B, X, decltype(j.t % resize<C>(n)), IMPL>(
          j.t % resize<C>(n), j.impl);
  }

  template <char DIM, size_t I, typename... TS>
  constexpr auto operator()(tuple_dim<DIM, I, TS...> t) {
    return make_tuple<DIM, I>(
        tuple_set_element<I>(t.ts, std::get<I>(t.ts) % resize<C>(n)));
  }
};

/* remove the fixed dimensions */
template <char C>
struct unfix_f {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr auto operator()(scalar_t<T> s) {
    return s;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t % unfix_f<C>()), IMPL>(
        d.t % unfix_f<C>(), d.impl);
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return d.t;
    else
      return fixed_dimension<DIM, decltype(d.t % unfix_f<C>())>(
          d.idx, d.t % unfix_f<C>());
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t % unfix_f<C>()), IMPL>(
        s.t % unfix_f<C>(), s.impl);
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t % unfix_f<C>()), IMPL>(
        j.t % unfix_f<C>(), j.impl);
  }

  template <char DIM, size_t I, typename... TS>
  constexpr auto operator()(tuple_dim<DIM, I, TS...> t) {
    return make_tuple<DIM, I>(
        tuple_set_element<I>(t.ts, std::get<I>(t.ts) % unfix_f<C>()));
  }
};

template <char C>
static constexpr unfix_f<C> unfix;

/* remove any fixed dimensions */
struct unfix_all_f {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr auto operator()(scalar_t<T> s) {
    return s;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t % unfix_all_f()), IMPL>(
        d.t % unfix_all_f(), d.impl);
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return d.t % unfix_all_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t % unfix_all_f()), IMPL>(
        s.t % unfix_all_f(), s.impl);
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t % unfix_all_f()), IMPL>(
        j.t % unfix_all_f(), j.impl);
  }

  template <char DIM, size_t I, typename... TS>
  constexpr auto operator()(tuple_dim<DIM, I, TS...> t) {
    return [&]<size_t... IS>(std::index_sequence<IS...>) {
      return make_tuple_dim<DIM, I>((std::get<IS>(t.ts) % unfix_all_f())...);
    }
    (std::make_index_sequence<sizeof...(TS)>());
  }
};

static constexpr unfix_all_f unfix_all;

/* fix an index in one dimension */
template <char C>
struct fix {
  size_t n;

  constexpr fix(size_t n) : n(n) {}

  template <typename K>
  constexpr auto operator()(K k) const {
    static_assert(has_dim_p<C, K>::get(), "dimension for fixing not found");
    return fixed_dimension<C, K>(n, k % unfix_f<C>());
  }
};

/* fix more things at once */
template <char C, char... CS>
struct fixs {
  size_t n;
  fixs<CS...> fs;

  template <typename... NS>
  constexpr fixs(size_t n, NS... ns) : n(n), fs(ns...) {}

  template <typename K>
  constexpr auto operator()(K k) const {
    return k % fix<C>(n) % fs;
  }
};

template <char C>
struct fixs<C> {
  size_t n;
  constexpr fixs(size_t n) : n(n) {}

  template <typename K>
  constexpr auto operator()(K k) const {
    return k % fix<C>(n);
  }
};

/* find a fixed dimension (TODO the limits and ranges) */

template <char C>
struct has_dimval_f {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr bool operator()(scalar_t<T> s) {
    return false;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr bool operator()(dimension<DIM, T, IMPL> d) {
    return d.t % has_dimval_f();
  }

  template <char DIM, typename T>
  constexpr bool operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return true;
    else
      return d.t % has_dimval_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr bool operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t % has_dimval_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr bool operator()(join2<A, B, X, T, IMPL> j) {
    return j.t % has_dimval_f();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr bool operator()(tuple_dim<DIM, I, TS...> t) {
    return std::get<I>(t.ts) % has_dimval_f();
  }
};

template <char C>
struct dimval_f {
  empty_struct_t empty_struct;

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return d.t % dimval_f();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return d.idx;
    else
      return d.t % dimval_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t % dimval_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t % dimval_f();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t) {
    return std::get<I>(t.ts) % dimval_f();
  }
};

/* set tuple "branch" */
template <char C, size_t FLD>
struct field_f {
  empty_struct_t empty_struct;

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t % field_f<C, FLD>()), IMPL>(
        d.t % field_f<C, FLD>(), d.impl);
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return fixed_dimension<DIM, decltype(d.t % field_f<C, FLD>())>(
        d.t % field_f<C, FLD>());
  }
  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t % field_f<C, FLD>()), IMPL>(
        s.t % field_f<C, FLD>(), s.impl);
  }
  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t % field_f<C, FLD>()), IMPL>(
        j.t % field_f<C, FLD>(), j.impl);
  }

  template <char DIM, size_t I, typename... TS>
  constexpr auto operator()(tuple_dim<DIM, I, TS...> t) {
    if constexpr (DIM == C)
      return make_tuple<DIM, FLD>(t.ts);
    else
      return make_tuple<DIM, I>(
          tuple_set_element<I>(t.ts, std::get<I>(t.ts) % field_f<C, FLD>()));
  }
};

template <char DIM, size_t FLD>
static constexpr field_f<DIM, FLD> field;

template <char C, size_t FLD, auto... ETC>
struct fields_f {
  empty_struct_t empty_struct;
  template <typename K>
  constexpr auto operator()(K k) {
    return k % field_f<C, FLD>() % fields_f<ETC...>();
  }
};

template <char C, size_t FLD>
struct fields_f<C, FLD> {
  empty_struct_t empty_struct;
  template <typename K>
  constexpr auto operator()(K k) {
    return k % field_f<C, FLD>();
  }
};

template <char DIM, size_t FLD, auto... ETC>
static constexpr fields_f<DIM, FLD, ETC...> fields;

// TODO has_field_p and pals

/* get the offset for the current fix (finally)! */
struct offset_f {
  empty_struct_t empty_struct;

  template <typename K>
  constexpr size_t operator()(K k) {
    return operator()(k, scalar_t<empty_struct_t>());
  }

  template <typename T, typename DIMS>
  constexpr size_t operator()(scalar_t<T> s, DIMS) {
    return 0;
  }

  template <char DIM, typename T, typename DIMS>
  constexpr size_t operator()(fixed_dimension<DIM, T> d, DIMS ds) {
    return operator()(d.t, fixed_dimension<DIM, DIMS>(d.idx, ds));
  }

  template <char DIM, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d, DIMS ds) {
    static_assert(IMPL::has_size, "dimension size unavailable");
    return ds % dimval_f<DIM>() * d.impl.step(d.t) + operator()(d.t, ds);
  }

  template <char A, char X, char Y, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s, DIMS ds) {
    auto [x, y] = s.impl(s.t, ds % dimval_f<A>());
    return operator()(s.t, fixed_dimension<X, fixed_dimension<Y, DIMS>>(
                               x, fixed_dimension<Y, DIMS>(y, ds)));
  }

  template <char A, char B, char X, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j, DIMS ds) {
    auto x = j.impl(j.t, ds % dimval_f<A>(), ds % dimval_f<B>());
    return operator()(j.t, fixed_dimension<X, DIMS>(x, ds));
  }

  template <char DIM, size_t I, typename... TS, typename DIMS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t, DIMS ds) {
    return size_f::tuple_size(tuple_slice<0, I>(t.ts)) + operator()(
                                                             std::get<I>(t.ts),
                                                             ds);
  }
};

static constexpr offset_f offset;

/* get a pointer relative to the data structure origin */
struct at {
  char* ptr;

  template <class PTR>
  constexpr at(PTR* ptr) : ptr(reinterpret_cast<char*>(ptr)) {}

  template <typename K>
  constexpr auto operator()(K k) const {
    return reinterpret_cast<typename K::scalar_type*>(ptr + k % offset_f());
  }
};

/* index in the array to get the offset (combined fix+offset) */
template <char C, char... CS>
struct idx {
  size_t n;
  idx<CS...> fs;

  template <typename... NS>
  idx(size_t n, NS... ns) : n(n), fs(ns...) {}

  template <typename K>
  constexpr size_t operator()(K k) const {
    return k % fix<C>(n) % fs;
  }
};

template <char C>
struct idx<C> {
  size_t n;

  idx(size_t n) : n(n) {}

  template <typename K>
  constexpr size_t operator()(K k) const {
    return k % fix<C>(n) % offset_f();
  }
};

/*
 * container implementations
 *
 * TODO: does padding belong here?
 */

template <size_t N>
struct array_impl {
  empty_struct_t empty_struct;

  constexpr array_impl() {}

  template <typename T>
  constexpr size_t step(T t) const {
    return t % size_f();
  }
  template <typename T>
  constexpr size_t size(T t) const {
    return N * (t % size_f());
  }

  static constexpr bool has_size = true;
};

struct vector_impl {
  size_t n;
  constexpr vector_impl(size_t n) : n(n) {}
  constexpr vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T>
  constexpr size_t step(T t) const {
    return t % size_f();
  }
  template <typename T>
  constexpr size_t size(T t) const {
    return n * (t % size_f());
  }

  static constexpr bool has_size = true;
};

struct vector_impl_unsized {
  empty_struct_t empty_struct;

  constexpr vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T>
  constexpr size_t step(T t) const {
    return t % size_f();
  }

  static constexpr bool has_size = false;
};

/*
 * Function implementations
 *
 * TODO: hilbert curves? is reverse z-order good for anything? block-merging
 * useful for anything?
 */

template <size_t N>
struct split2_bitblock {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr std::pair<size_t, size_t> operator()(T t, size_t idx) {
    return {idx >> N, idx & ((size_t(1) << N) - 1)};
  }
};

template <size_t N>
struct split2_divmod {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr std::pair<size_t, size_t> operator()(T t, size_t idx) {
    return {idx / N, idx % N};
  }
};

struct join2_zorder32 {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr size_t operator()(T t, size_t idx1, size_t idx2) {
    idx1 &= 0xffffffff;
    idx1 = (idx1 ^ (idx1 << 16)) & 0x0000ffff0000ffff;
    idx1 = (idx1 ^ (idx1 << 8)) & 0x00ff00ff00ff00ff;
    idx1 = (idx1 ^ (idx1 << 4)) & 0x0f0f0f0f0f0f0f0f;
    idx1 = (idx1 ^ (idx1 << 2)) & 0x3333333333333333;
    idx1 = (idx1 ^ (idx1 << 1)) & 0x5555555555555555;
    idx2 &= 0xffffffff;
    idx2 = (idx2 ^ (idx2 << 16)) & 0x0000ffff0000ffff;
    idx2 = (idx2 ^ (idx2 << 8)) & 0x00ff00ff00ff00ff;
    idx2 = (idx2 ^ (idx2 << 4)) & 0x0f0f0f0f0f0f0f0f;
    idx2 = (idx2 ^ (idx2 << 2)) & 0x3333333333333333;
    idx2 = (idx2 ^ (idx2 << 1)) & 0x5555555555555555;
    return idx1 | (idx2 << 1);
  }
};

/*
 * type assembly shortcuts
 */

template <typename T>
static constexpr scalar_t<T> scalar;

template <char C, size_t N>
struct array_part {
  empty_struct_t empty_struct;
};

template <typename K, char C, size_t N>
constexpr auto operator^(K k, array_part<C, N> a) {
  return dimension<C, K, array_impl<N>>(k, array_impl<N>());
}

template <char C, size_t N>
static constexpr array_part<C, N> array;

template <char C>
struct unsized_vector_part {
  empty_struct_t empty_struct;
};

template <typename K, char C>
constexpr auto operator^(K k, unsized_vector_part<C> v) {
  return dimension<C, K, vector_impl_unsized>(k, vector_impl_unsized());
}

template <char C>
static constexpr unsized_vector_part<C> vector;

template <char C>
struct vector_sized {
  size_t n;

  constexpr vector_sized(size_t n) : n(n) {}
};

template <typename K, char C>
constexpr auto operator^(K k, vector_sized<C> v) {
  return dimension<C, K, vector_impl>(k, vector_impl(v.n));
}

/* semantic problem: splits/joins read "forward" in types but "backwards" in
 * the ^ notation, which isn't really much useful for guessing which of the
 * variables are taken from right and get created on the left. The naming is
 * thus different to prevent ambiguity:
 *
 * - zorder12: takes 2 dims on the right, produces a single dimension
 * - bitblock21: takes a dimension and produces 2 by blocks
 */
template <char X, char A, char B>
struct zorder12_part {
  empty_struct_t empty_struct;
};

template <typename K, char X, char A, char B>
constexpr auto operator^(K k, zorder12_part<X, A, B>) {
  return join2<A, B, X, K, join2_zorder32>(k);
}

template <char X, char A, char B>
static constexpr zorder12_part<X, A, B> zorder12;

template <char X, char Y, size_t N, char A>
struct bitblock21_part {
  empty_struct_t empty_struct;
};

template <typename K, char X, char Y, size_t N, char A>
constexpr auto operator^(K k, bitblock21_part<X, Y, N, A>) {
  return split2<A, X, Y, K, split2_bitblock<N>>(k);
}

template <char X, char Y, size_t N, char A>
static constexpr bitblock21_part<X, Y, N, A> bitblock21;

template <char X, char Y, size_t N, char A>
struct block21_part {
  empty_struct_t empty_struct;
};

template <typename K, char X, char Y, size_t N, char A>
constexpr auto operator^(K k, block21_part<X, Y, N, A>) {
  return split2<A, X, Y, K, split2_divmod<N>>(k);
}

template <char X, char Y, size_t N, char A>
static constexpr block21_part<X, Y, N, A> block21;

template <char DIM>
struct empty_tuple_dim {
  empty_struct_t empty_struct;
};

template <char DIM, typename K>
constexpr auto operator*(empty_tuple_dim<DIM> t, K k) {
  return tuple_dim<DIM, 0, K>(k);
}

template <char DIM, size_t I, typename... TS, typename K>
constexpr auto operator*(tuple_dim<DIM, I, TS...> t, K k) {
  return tuple_dim<DIM, I, TS..., K>(std::tuple_cat(t.ts, std::tuple<K>(k)));
}

template <char DIM>
static constexpr empty_tuple_dim<DIM> tuple;

/*
 * OOP interface
 */

template<typename>
struct void_template;

template<class... Ts>
struct first_type;

template<class T, class... Ts>
struct first_type<T, Ts...> {
    using type = T;
};

template<typename... Ts>
using first_type_t = typename first_type<Ts...>::type;

template<typename T, template <typename> class A = void_template>
struct wrapper;

template<template <typename> class A>
struct _wrapper_type {
  template<typename T>
  using type = A<T>;
};

template<>
struct _wrapper_type<void_template> {
template<typename T>
  using type = wrapper<T, void_template>;
};

template<typename T, template <typename> class A>
struct wrapper {
  T t;

  template<typename T2>
  using wrapper_type = typename _wrapper_type<A>::template type<T2>;

  constexpr wrapper(T t) : t{t} {}

  template<typename = void>
  constexpr T unwrap() const {
    return t;
  }

  template<typename = void>
  constexpr wrapper_type<size_t> size() const {
    return size_f()(t);
  }

  template<typename = void>
  constexpr wrapper_type<size_t> depth() const {
    return depth_f()(t);
  }

  template<char C>
  constexpr wrapper_type<size_t> level() const {
    return level_f<C>()(t);
  }

  template<char C>
  constexpr wrapper_type<bool> has_dim() const {
    return has_dim_f<C>()(t);
  }

  template<char... CS>
  constexpr wrapper_type<bool> has_dims() const {
    return has_dims_f<CS...>()(t);
  }

  template<char C>
  constexpr auto resize(size_t n) const -> wrapper_type<decltype(arrr::resize<C>(n)(t))> {
    return arrr::resize<C>(n)(t);
  }

  template<char C>
  constexpr wrapper_type<decltype(unfix_f<C>()(t))> unfix() const {
    return unfix_f<C>()(t);
  }

  template<typename _ = void>
  constexpr wrapper_type<std::invoke_result_t<unfix_all_f, first_type_t<T, _>>> unfix_all() const {
    return unfix_all_f()(t);
  }

  template<char C>
  constexpr auto fix(size_t n) const -> wrapper_type<decltype(arrr::fix<C>::operator()(t))> {
    return arrr::fix<C>(n)(t);
  }

  //TODO the template parameter here is problematic, fix it with an extra struct. :D
  template<char... CS>
  constexpr auto fixs(int n, int m) const -> wrapper_type<decltype(arrr::fixs<CS...>(n, m)(t))> {
    return arrr::fixs<CS...>(n, m)(t);
  }

  template<char C, size_t FLD>
  constexpr wrapper_type<decltype(field_f<C, FLD>()(t))> field() const {
    return field_f<C, FLD>()(t);
  }

  template<auto... ETC>
  constexpr wrapper_type<decltype(fields_f<ETC...>()(t))> fields() const {
    return fields_f<ETC...>()(t);
  }

  template<typename = void>
  constexpr wrapper_type<size_t> offset() const {
    return offset_f()(t);
  }

  //TODO const variant
  template<class PTR>
  constexpr auto at(PTR* ptr) const -> wrapper_type<std::invoke_result_t<typename arrr::at,first_type_t<T, PTR>>> {
    return at(ptr)(t);
  }

  //TODO idx (same problem as with fixs)

  template<typename TT>
  static constexpr scalar_t<TT> scalar() {
    return scalar_t<TT>();
  }

  //TODO the same for array etc.
};

template<template<typename> class = wrapper>
struct wrapper_policy {

};

template<typename T, template<typename> class W = wrapper>
constexpr auto wrap(T t, wrapper_policy<W> = wrapper_policy<W>{}) {
  return W<T>(t);
}

}  // namespace arrr

/*
 * TESTS
 */

#include <iostream>
#include <typeinfo>
using namespace arrr;
using std::cout;
using std::endl;



template<typename T>
struct my_wrapper : arrr::wrapper<T, my_wrapper> {
  constexpr my_wrapper(T t) : arrr::wrapper<T, my_wrapper>{t} {}


  template<typename = void>
  constexpr my_wrapper<size_t> offset() const {
    std::cout << "calling 'offset' on " << typeid(T).name() << std::endl;
    return offset_f()(arrr::wrapper<T, my_wrapper>::t);
  }

};


int main() {
  auto a = wrap(scalar<float> ^ vector<'y'> ^ array<'x', 20>);
  auto b = wrap(scalar<float> ^ vector<'y'> ^ array<'x', 20>, wrapper_policy<my_wrapper>{});

  auto as = a.resize<'y'>(30);  // TODO rest sized vectors
  auto bs = b.resize<'y'>(30);  // TODO rest sized vectors

  std::cout << as.fixs<'x','y'>(10, 10).offset().unwrap() << endl;
  std::cout << bs.fixs<'x','y'>(10, 10).offset().unwrap() << endl;
}
