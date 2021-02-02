#include <cassert>
#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace arrr {

template <typename T, bool empty = std::is_empty_v<T>>
struct emptyable_t;

template <typename T>
struct emptyable_t<T, false> {
  using type = T;
  T t_;
  constexpr T t() const { return t_; }
  constexpr emptyable_t() {}
  constexpr emptyable_t(T t) : t_(t) {}
};

template <typename T>
struct emptyable_t<T, true> {
  using type = T;
  constexpr T t() const { return T(); }
  constexpr emptyable_t() {}
  constexpr emptyable_t(T){};
};

/* this emptyable_* is here twice just because it needs a different name so
 * that both can get inherited into another struct together; normally it would
 * be a clear newtype */
template <typename IMPL, bool empty_impl=std::is_empty_v<IMPL>>
struct emptyable_impl;

template <typename IMPL>
struct emptyable_impl<IMPL, false> {
  IMPL impl_;
  constexpr IMPL impl() const { return impl_; }
  constexpr emptyable_impl() {}
  constexpr emptyable_impl(IMPL impl) : impl_(impl) {}
};

template <typename IMPL>
struct emptyable_impl<IMPL, true> {
  constexpr IMPL impl() const { return IMPL(); }
  constexpr emptyable_impl() {}
  constexpr emptyable_impl(IMPL) {}
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
  using scalar_type = T;
  constexpr scalar_t() {}
};

/* 1-dimensions with char name */
template <char DIM, typename T, typename IMPL>
struct dimension : public emptyable_t<T>, public emptyable_impl<IMPL> {
  using scalar_type = typename T::scalar_type;

  constexpr dimension() {}

  template <typename... Args>
  constexpr dimension(T t, Args... a) : emptyable_t<T>(t), emptyable_impl<IMPL>(IMPL(a...)) {}
};

/* dimension index fix */
template <char DIM, typename T>
struct fixed_dimension : public emptyable_t<T>{
  size_t idx;

  using scalar_type = typename T::scalar_type;

  constexpr fixed_dimension() {}

  template <typename... Args>
  constexpr fixed_dimension(size_t idx, Args... a) : idx(idx), emptyable_t<T>(a...) {}
};

/* converts a single dimension to 2 */
template <char A, char X, char Y, typename T, typename IMPL>
struct split2 : public emptyable_t<T>, public emptyable_impl<IMPL>{
  using scalar_type = typename T::scalar_type;

  constexpr split2() {}
  template <typename... Args>
  constexpr split2(T t, Args... a) : emptyable_t<T>(t), emptyable_impl<IMPL>(IMPL(a...)) {}
};

/* merges 2 dimensions into 1 */
template <char A, char B, char X, typename T, typename IMPL>
struct join2 : public emptyable_t<T>, public emptyable_impl<IMPL>{
  using scalar_type = typename T::scalar_type;

  constexpr join2() {}
  template <typename... Args>
  constexpr join2(T t, Args... a)  : emptyable_t<T>(t), emptyable_impl<IMPL>(IMPL(a...)) {}
};

/* tuple machinery (fun fact: std::tuple checks std::is_empty automagically) */
template <char DIM, size_t I, typename... TS>
struct tuple_dim {
  using TUP = std::tuple<TS...>;
  TUP ts;

  template <size_t J>
  using tuple_element = typename std::tuple_element<J, TUP>::type;

  using current_type = tuple_element<I>;
  using scalar_type = typename current_type::scalar_type;

  constexpr tuple_dim() {}
  constexpr tuple_dim(TS... ts) : ts(ts...) {}
  constexpr tuple_dim(TUP tup) : ts(tup) {}
};

/* tuple type modifications; thanks @sebrockm on stackoverflow */
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
  template <typename T>
  constexpr size_t operator()(scalar_t<T> s) {
    return sizeof(T);
  }

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return d.impl().size(d.t());
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t() % size_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t() % size_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t() % size_f();
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
  template <typename T>
  constexpr size_t operator()(scalar_t<T> s) {
    return 0;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return 1 + d.t() % depth_f();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t() % depth_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t() % depth_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t() % depth_f();
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
  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return d.t() % depth_f();
    else
      return d.t() % level_f<C>();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    return d.t() % level_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    if constexpr (A == C)
      return s.t() % depth_f();
    else
      return s.t() % level_f<C>();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    if constexpr (A == C || B == C)
      return j.t() % depth_f();
    else
      return j.t() % level_f<C>();
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
  using TI = typename tuple_dim<DIM, I, TS...>::current_type;
  static constexpr bool get() {
    return has_dim_p<C, TI>::get();  // TRICKY: tuple "dimensions" are distinct
                                     // from normal dimensions
  }
};

/* value interface for the above */
template <char C>
struct has_dim_f {
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
  template <typename K>
  constexpr bool operator()(K k) {
    return k % has_dim_f<C>() && k % has_dims_f<CS...>();
  }
};

template <char C>
struct has_dims_f<C> {
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
  size_t n;

  constexpr resize(size_t n) : n(n) {}

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return dimension<DIM, T, decltype(d.impl().resize(n))>(d.t(),
                                                           d.impl().resize(n));
    else
      return dimension<DIM, decltype(d.t() % resize<C>(n)), IMPL>(
          d.t() % resize<C>(n), d.impl());
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return d.t() % resize<C>(n);  // FIXME this removes the fix, is that right?
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    if constexpr (A == C)
      return split2<A, X, Y, T, decltype(s.impl().resize(n))>(s.t(),
                                                            s.impl().resize(n));
    else
      return split2<A, X, Y, decltype(s.t % resize<C>(n)), IMPL>(
          s.t() % resize<C>(n), s.impl());
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    if constexpr (A == C)
      return join2<A, B, X, T, decltype(j.impl().resize1(n))>(j.t,
                                                            j.impl().resize1(n));
    else if constexpr (B == C)
      return join2<A, B, X, T, decltype(j.impl().resize2(n))>(j.t,
                                                            j.impl().resize2(n));
    else
      return join2<A, B, X, decltype(j.t % resize<C>(n)), IMPL>(
          j.t % resize<C>(n), j.impl());
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
  template <typename T>
  constexpr auto operator()(scalar_t<T> s) {
    return s;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t() % unfix_f<C>()), IMPL>(
        d.t() % unfix_f<C>(), d.impl());
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return d.t();
    else
      return fixed_dimension<DIM, decltype(d.t() % unfix_f<C>())>(
          d.idx, d.t() % unfix_f<C>());
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t() % unfix_f<C>()), IMPL>(
        s.t() % unfix_f<C>(), s.impl());
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t() % unfix_f<C>()), IMPL>(
        j.t() % unfix_f<C>(), j.impl());
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
  template <typename T>
  constexpr auto operator()(scalar_t<T> s) {
    return s;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t() % unfix_all_f()), IMPL>(
        d.t() % unfix_all_f(), d.impl());
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return d.t() % unfix_all_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t() % unfix_all_f()), IMPL>(
        s.t() % unfix_all_f(), s.impl());
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t() % unfix_all_f()), IMPL>(
        j.t() % unfix_all_f(), j.impl());
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
  template <typename T>
  constexpr bool operator()(scalar_t<T> s) {
    return false;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr bool operator()(dimension<DIM, T, IMPL> d) {
    return d.t() % has_dimval_f();
  }

  template <char DIM, typename T>
  constexpr bool operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return true;
    else
      return d.t() % has_dimval_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr bool operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t() % has_dimval_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr bool operator()(join2<A, B, X, T, IMPL> j) {
    return j.t() % has_dimval_f();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr bool operator()(tuple_dim<DIM, I, TS...> t) {
    return std::get<I>(t.ts) % has_dimval_f();
  }
};

template <char C>
struct dimval_f {
  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return d.t() % dimval_f();
  }

  template <char DIM, typename T>
  constexpr size_t operator()(fixed_dimension<DIM, T> d) {
    if constexpr (DIM == C)
      return d.idx;
    else
      return d.t() % dimval_f();
  }

  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s) {
    return s.t() % dimval_f();
  }

  template <char A, char B, char X, typename T, typename IMPL>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j) {
    return j.t() % dimval_f();
  }

  template <char DIM, size_t I, typename... TS>
  constexpr size_t operator()(tuple_dim<DIM, I, TS...> t) {
    return std::get<I>(t.ts) % dimval_f();
  }
};

/* set tuple "branch" */
template <char C, size_t FLD>
struct field_f {
  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t() % field_f<C, FLD>()), IMPL>(
        d.t() % field_f<C, FLD>(), d.impl());
  }

  template <char DIM, typename T>
  constexpr auto operator()(fixed_dimension<DIM, T> d) {
    return fixed_dimension<DIM, decltype(d.t() % field_f<C, FLD>())>(
        d.t() % field_f<C, FLD>());
  }
  template <char A, char X, char Y, typename T, typename IMPL>
  constexpr auto operator()(split2<A, X, Y, T, IMPL> s) {
    return split2<A, X, Y, decltype(s.t() % field_f<C, FLD>()), IMPL>(
        s.t() % field_f<C, FLD>(), s.impl());
  }
  template <char A, char B, char X, typename T, typename IMPL>
  constexpr auto operator()(join2<A, B, X, T, IMPL> j) {
    return join2<A, B, X, decltype(j.t() % field_f<C, FLD>()), IMPL>(
        j.t() % field_f<C, FLD>(), j.impl());
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
  template <typename K>
  constexpr auto operator()(K k) {
    return k % field_f<C, FLD>() % fields_f<ETC...>();
  }
};

template <char C, size_t FLD>
struct fields_f<C, FLD> {
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
  template <typename K>
  constexpr size_t operator()(K k) {
    return operator()(k, scalar_t<void>());
  }

  template <typename T, typename DIMS>
  constexpr size_t operator()(scalar_t<T> s, DIMS) {
    return 0;
  }

  template <char DIM, typename T, typename DIMS>
  constexpr size_t operator()(fixed_dimension<DIM, T> d, DIMS ds) {
    return operator()(d.t(), fixed_dimension<DIM, DIMS>(d.idx, ds));
  }

  template <char DIM, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d, DIMS ds) {
    static_assert(IMPL::has_size, "dimension size unavailable");
    return ds % dimval_f<DIM>() * d.impl().step(d.t()) + operator()(d.t(), ds);
  }

  template <char A, char X, char Y, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(split2<A, X, Y, T, IMPL> s, DIMS ds) {
    auto [x, y] = s.impl()(s.t(), ds % dimval_f<A>());
    return operator()(s.t(), fixed_dimension<X, fixed_dimension<Y, DIMS>>(
                               x, fixed_dimension<Y, DIMS>(y, ds)));
  }

  template <char A, char B, char X, typename T, typename IMPL, typename DIMS>
  constexpr size_t operator()(join2<A, B, X, T, IMPL> j, DIMS ds) {
    auto x = j.impl()(j.t(), ds % dimval_f<A>(), ds % dimval_f<B>());
    return operator()(j.t(), fixed_dimension<X, DIMS>(x, ds));
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
 */

template <size_t N>
struct array_impl {
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
 * TODO: is reverse z-order and block-merging good for anything?
 */

template <size_t N>
struct split2_bitblock {
  template <typename T>
  constexpr std::pair<size_t, size_t> operator()(T t, size_t idx) {
    return {idx >> N, idx & ((size_t(1) << N) - 1)};
  }
};

template <size_t N>
struct split2_divmod {
  template <typename T>
  constexpr std::pair<size_t, size_t> operator()(T t, size_t idx) {
    return {idx / N, idx % N};
  }
};

struct join2_zorder32 {
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
struct array_part {};

template <typename K, char C, size_t N>
constexpr auto operator^(K k, array_part<C, N> a) {
  return dimension<C, K, array_impl<N>>(k, array_impl<N>());
}

template <char C, size_t N>
static constexpr array_part<C, N> array;

template <char C>
struct unsized_vector_part {};

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
struct zorder12_part {};

template <typename K, char X, char A, char B>
constexpr auto operator^(K k, zorder12_part<X, A, B>) {
  return join2<A, B, X, K, join2_zorder32>(k);
}

template <char X, char A, char B>
static constexpr zorder12_part<X, A, B> zorder12;

template <char X, char Y, size_t N, char A>
struct bitblock21_part {};

template <typename K, char X, char Y, size_t N, char A>
constexpr auto operator^(K k, bitblock21_part<X, Y, N, A>) {
  return split2<A, X, Y, K, split2_bitblock<N>>(k);
}

template <char X, char Y, size_t N, char A>
static constexpr bitblock21_part<X, Y, N, A> bitblock21;

template <char X, char Y, size_t N, char A>
struct block21_part {};

template <typename K, char X, char Y, size_t N, char A>
constexpr auto operator^(K k, block21_part<X, Y, N, A>) {
  return split2<A, X, Y, K, split2_divmod<N>>(k);
}

template <char X, char Y, size_t N, char A>
static constexpr block21_part<X, Y, N, A> block21;

template <char DIM>
struct empty_tuple_dim {};

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
};  // namespace arrr

/*
 * TESTS
 */

#include <iostream>
#include <typeinfo>
using namespace arrr;
using std::cout;
using std::endl;

int main() {
  auto a = scalar<float> ^ vector<'y'> ^ array<'x', 20>;
  cout << "has_dim tests:" << endl;
  cout << a % has_dim<'x'> << endl;
  cout << a % has_dim<'y'> << endl;
  cout << a % has_dim<'z'> << endl;
  cout << a % has_dims<'x'> << endl;
  cout << a % has_dims<'x', 'y'> << endl;
  cout << a % has_dims<'x', 'y', 'z'> << endl;

  auto as = a % resize<'y'>(30);  // TODO rest sized vectors

  // auto as = scalar<float> ^ vector_sized<'y'>(30) ^ array<'x', 20>;
  cout << "more has_dim tests:" << endl;
  cout << as % has_dim<'x'> << endl;
  cout << as % has_dim<'y'> << endl;
  cout << as % has_dim<'z'> << endl;
  cout << as % has_dims<'x'> << endl;
  cout << as % has_dims<'x', 'y'> << endl;
  cout << as % has_dims<'x', 'y', 'z'> << endl;

  cout << "sizes:" << endl;
  cout << sizeof(a) << endl;
  cout << sizeof(as) << endl;
  cout << typeid(a).name() << endl;
  cout << typeid(as).name() << endl;

  cout << "levels/depths:" << endl;
  cout << a % depth << endl;
  cout << a % level<'x'> << endl;
  cout << as % level<'x'> << endl;
  cout << a % level<'y'> << endl;
  cout << as % level<'y'> << endl;

  auto asf = as % fix<'x'>(10);
  auto asff = asf % fix<'y'>(10);
  cout << "fixed index:" << asff % offset << endl;
  cout << sizeof(asf) << endl;
  cout << typeid(asf).name() << endl;
  cout << sizeof(asff) << endl;
  cout << typeid(asff).name() << endl;

  auto asffu = asff % unfix<'x'>;
  cout << sizeof(asffu) << endl;
  cout << typeid(asffu).name() << endl;

  auto asffua = asff % unfix_all;
  cout << sizeof(asffua) << endl;
  cout << typeid(asffua).name() << endl;

  cout << "multifix:" << endl;
  cout << as % fixs<'x', 'y'>(3, 5) % offset << endl;
  cout << as % idx<'x', 'y'>(3, 5) << endl;
  for (size_t i = 0; i < 10; ++i) cout << as % idx<'y', 'x'>(3, i) << endl;

  float x[20 * 30];
  cout << "actual data access:" << endl;
  cout << typeid(asff % at(x)).name() << endl;
  cout << asff % at(x) << endl;
  cout << "uninitialized read: " << *(asff % at(x)) << endl;
  *(asff % at(x)) = 10;
  cout << *(asff % at(x)) << endl;

  auto spl = scalar<char> ^ array<'X', 5> ^ array<'x', 4> ^
             bitblock21<'X', 'x', 2, 'x'>;

  cout << "split test:" << endl;
  for (size_t i = 0; i < 20; ++i) cout << spl % idx<'x'>(i) << endl;

  auto joi = scalar<char> ^ array<'z', 16> ^ zorder12<'z', 'x', 'y'>;

  cout << "join test:" << endl;
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) cout << joi % idx<'x', 'y'>(x, y) << '\t';
    cout << endl;
  }

  // TODO: test the normal modulo block
  auto magic = scalar<char> ^ array<'y', 4> ^ array<'x', 4> ^ array<'z', 16> ^
               zorder12<'z', 'Y', 'X'> ^ bitblock21<'Y', 'y', 2, 'y'> ^
               bitblock21<'X', 'x', 2, 'x'>;

  cout << "brutal magic test:" << endl;
  for (size_t x = 0; x < 16; ++x) {
    for (size_t y = 0; y < 16; ++y) cout << magic % idx<'x', 'y'>(x, y) << '\t';
    cout << endl;
  }

  cout << "tuple test:" << endl;
  auto tup1 = tuple<'a'> * scalar<float>;
  auto tup2 =
      tuple<'a'> * (scalar<float> ^ vector<'i'>)*scalar<int> ^ array<'j', 300>;

  cout << sizeof(tup1) << endl;
  cout << sizeof(tup2) << endl;
  cout << sizeof(tup2 % resize<'i'>(20)) << endl;
  cout << typeid(tup1).name() << endl;
  cout << typeid(tup2).name() << endl;
  cout << tup1 % size << endl;
  cout << tup2 % resize<'i'>(20) % size << endl;
  cout << tup2 % resize<'i'>(21) % size << endl;
  cout << tup2 % resize<'i'>(22) % size << endl;
  cout << sizeof(tup2) << endl;
  cout << sizeof(tup2 % fix<'i'>(123)) << endl;
  cout << sizeof(tup2 % fix<'i'>(123) % unfix_all) << endl;
  cout << tup2 % has_dim<'j'> << endl;
  cout << (tuple<'a'> * (scalar<float> ^ vector<'i'>)) % has_dim<'i'> << endl;

  cout << "tuple offsets:" << endl;
  cout << typeid(tup2 % field<'a', 0>).name() << endl;
  cout << typeid(tup2 % field<'a', 1>).name() << endl;
  auto t2s = tup2 % resize<'i'>(500);
  cout << t2s % field<'a', 0> % idx<'i', 'j'>(1, 2) << endl;
  cout << t2s % field<'a', 1> % idx<'j'>(1) << endl;

  auto t3 = tuple<'a'> * (tuple<'b'> * scalar<char> * scalar<int>)*(
                             tuple<'c'> * scalar<float> * scalar<double>);
  cout << t3 % fields<'a', 0, 'b', 0> % offset << endl;
  cout << t3 % fields<'a', 0, 'b', 1> % offset
       << endl;  // this gonna need some serious padding
  cout << t3 % fields<'a', 1, 'c', 1> % offset << endl;

  return 0;
}
