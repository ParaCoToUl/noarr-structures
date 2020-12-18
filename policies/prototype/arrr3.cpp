#include <cassert>
#include <cstddef>

namespace arrr {

/* recursion helper */
template <char... CS>
constexpr bool dims_empty() {
  return false;
}
template <>
constexpr bool dims_empty<>() {
  return true;
}

/* this has size 0, empty structs have otherwise size 1 */
struct empty_struct_t {
  int empty_struct_dummy[0] = {};
};

/* polymorphic application (specialize this for kinds) */
template <typename F, typename K>
constexpr auto operator%(K k, F f) {
  return f(k);
}

/*
 * Basic type kinds!
 */

/* scalars */
template <typename T>
struct scalar {
  empty_struct_t empty_struct;

  constexpr scalar() {}
  constexpr scalar(const scalar&){}

  constexpr size_t step() const { return sizeof(T); }
  constexpr size_t size() const { return sizeof(T); }
};

/* 1-dimensions with char name */
template <char DIM, typename T, typename IMPL>
struct dimension {
  T t;
  IMPL impl;

  constexpr dimension() {}

  template <typename... Args>
  constexpr dimension(T t, Args... a) : t(t), impl(a...) {}
  constexpr dimension(const dimension &d) : t(d.t), impl(d.impl) {}

  constexpr size_t step() const { return impl.step(t); }
  constexpr size_t size() const { return impl.size(t); }

  constexpr T get_t() const { return t; }
};

template <typename D>
struct fixed_dimension {
  size_t idx;
  D dim;

  constexpr fixed_dimension() {}
  constexpr fixed_dimension(const fixed_dimension&d) : idx(d.idx), dim(d.dim) {}

  template <typename... Args>
  constexpr fixed_dimension(size_t idx, Args... a) : idx(idx), dim(a...) {}

  constexpr size_t step() const { return dim.step(); }
  constexpr size_t size() const { return dim.size(); }
};

/* TODO (far future) extra kinds:
 * - tuple
 * - transform12 (eats a dim, creates 2, e.g. for Z-order expansion or for block
 *   splits)
 * - transform21 (eats 2 dims, creates 1, e.g. for Z-order contraction or
 *   triangles/symmetries)
 */

/*
 * Functions on type kinds!
 */

/* demo: total layers of construction all the way to a scalar */
struct depth_f {
  empty_struct_t empty_struct;
  template <typename T>
  constexpr size_t operator()(scalar<T> s) {
    return 0;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    return 1 + d.t % depth_f();
  }

  template <typename D>
  constexpr size_t operator()(fixed_dimension<D> d) {
    return 0 + d.dim % depth_f();
  }
};

/* reverse depth -- construction layers between the index and a base scalar */
template <char C>
struct level_f {
  empty_struct_t empty_struct;
  template <char DIM, typename T, typename IMPL>
  constexpr size_t operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C) {
      return d.t % depth_f();
    } else
      return d.t % level_f<C>();
  }

  template <typename D>
  constexpr size_t operator()(fixed_dimension<D> d) {
    return d.dim % level_f();
  }
};

/* does the structure have a character index? */
template <char C>
struct has_dim_f {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr bool operator()(scalar<T> s) {
    return false;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr bool operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C) {
      static_assert(!(d.get_t() % has_dim_f<C>()), "redundant index");
      return true;
    } else
      return d.t % has_dim_f<C>();
  }

  template <typename D>
  constexpr bool operator()(fixed_dimension<D> d) {
    return d.t % has_dim_f<C>(d.dim);
  }
};

/* does the structure have a list of character indexes? */
template <char C, char... CS>
struct has_dims_f {
  empty_struct_t empty_struct;
  template <typename T>
  constexpr bool operator()(scalar<T> s) {
    return false;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr bool operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (dims_empty<CS...>())
      return d % has_dim_f<C>();
    else
      return d % has_dim_f<C>() && d % has_dims_f<CS...>();
  }

  template <typename D>
  constexpr bool operator()(fixed_dimension<D> d) {
    return d.dim % has_dims_f<CS...>();
  }
};

/* resize unsized_vectors to sized vectors
 * TODO: unsize, unsize_all */
template <char C>
struct resize {
  empty_struct_t empty_struct;
  size_t n;

  resize(size_t n) : n(n) {}

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return dimension<DIM, T, decltype(d.impl.resize(n))>(d.t,
                                                           d.impl.resize(n));
    else
      return dimension<DIM, decltype(d.t % resize<C>(n)), IMPL>(
          d.t % resize<C>(n), d.impl);
  }

  template <typename D>
  constexpr auto operator()(fixed_dimension<D> d) {
    return d.dim % resize<C>(n);
  }
};

/* fix an index in one dimension*/
template <char C>
struct fix {
  size_t n;

  fix(size_t n) : n(n) {}

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    if constexpr (DIM == C)
      return fixed_dimension<dimension<DIM, T, IMPL>>(n, d);
    else
      return dimension<DIM, decltype(d.t % fix<C>(n)), IMPL>(d.t % fix<C>(n),
                                                             d.impl);
  }
  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(fixed_dimension<dimension<DIM, T, IMPL>> d) {
    if constexpr (DIM == C)
      return fixed_dimension<dimension<DIM, T, IMPL>>(
          n, dimension<DIM, T, IMPL>(d.dim.t, d.dim.impl));
    else
      return fixed_dimension<
          dimension<DIM, decltype(d.dim.t % fix<C>(n)), IMPL>>(
          d.idx, dimension<DIM, decltype(d.dim.t % fix<C>(n)), IMPL>(
                     d.dim.t % fix<C>(n), d.dim.impl));
  }
};

/* fix more things at once */
template<char C, char... CS>
struct fixs {
  size_t n;
  fixs<CS...> fs;

  template<typename... NS>
  fixs(size_t n, NS... ns) : n(n), fs(ns...) {}

  template<typename K>
  constexpr auto operator()(K k) {
    return k % fix<C>(n) % fs;
  }
};

template<char C>
struct fixs<C>
{
  size_t n;
  fixs(size_t n) : n(n) {}

  template<typename K>
  constexpr auto operator()(K k) {
    return k % fix<C>(n);
  }
};

/* remove the fixed dimensions */
template <char C>
struct unfix_f {
  empty_struct_t empty_struct;

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    static_assert(DIM != C, "unfixing an unfixed dimension");
    return dimension<DIM, decltype(d.t % unfix_f<C>()), IMPL>(
        d.t % unfix_f<C>(), d.impl);
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(fixed_dimension<dimension<DIM, T, IMPL>> d) {
    if constexpr (DIM == C)
      return d.dim;
    else
      return dimension<DIM, T, IMPL>(d.idx, d.dim % unfix_f<C>());
  }
};

/* remove any fixed dimensions */
struct unfix_all_f {
  empty_struct_t empty_struct;

  template<typename T>
  constexpr auto operator()(scalar<T> s) {
    return s;
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(dimension<DIM, T, IMPL> d) {
    return dimension<DIM, decltype(d.t % unfix_all_f()), IMPL>(
        d.t % unfix_all_f(), d.impl);
  }

  template <char DIM, typename T, typename IMPL>
  constexpr auto operator()(fixed_dimension<dimension<DIM, T, IMPL>> d) {
    return d.dim % unfix_all_f();
  }
};

/* get the offset for the current fix (finally)! */
struct offset_f {
  empty_struct_t empty_struct;

  template<typename T>
  constexpr size_t operator()(scalar<T> s) {
    return 0;
  }

  template<char DIM, typename T, typename IMPL>
  constexpr size_t operator()(fixed_dimension<dimension<DIM,T,IMPL>> d) {
    return d.idx * d.dim.step() + d.dim.t % offset_f();
  }
};

/* index in the array to get the offset (combined fix+offset) */
template<char C, char... CS>
struct idx {
  size_t n;
  idx<CS...> fs;

  template<typename...NS>
  idx(size_t n, NS... ns) : n(n), fs(ns...) {}

  template<typename K>
  constexpr size_t operator()(K k) {
    return k % fix<C>(n) % fs;
  }
};

template<char C>
struct idx<C> {
  size_t n;

  idx(size_t n) : n(n) {}

  template<typename K>
  constexpr size_t operator()(K k) {
    return k % fix<C>(n) % offset_f();
  }
};

/* function shortcuts */
static constexpr depth_f depth;

template <char C>
static constexpr level_f<C> level;

template <char C>
static constexpr has_dim_f<C> has_dim;

template <char... CS>
static constexpr has_dims_f<CS...> has_dims;

template <char C>
static constexpr unfix_f<C> unfix;
static constexpr unfix_all_f unfix_all;

static constexpr offset_f offset;

/*
 * container implementations
 */

template <size_t N>
struct array_impl {
  empty_struct_t empty_struct;

  template <typename T>
  constexpr size_t step(T t) const {
    return t.size();
  }
  template <typename T>
  constexpr size_t size(T t) const {
    return N * step(t);
  }
  template <typename T>
  constexpr size_t index(size_t i) const {
    return i;
  }
};

struct vector_impl {
  size_t n;
  vector_impl(size_t n) : n(n) {}
  vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T>
  constexpr size_t step(T t) const {
    return t.step();
  }
  template <typename T>
  constexpr size_t size(T t) const {
    return n * step(t);
  }
  template <typename T>
  constexpr size_t index(size_t i) const {
    return i;
  }
};

struct vector_impl_unsized {
  empty_struct_t empty_struct;

  vector_impl resize(size_t n) const { return vector_impl(n); }
  template <typename T>
  constexpr size_t step(T t) const {
    return t.step();
  }
  template <typename T>
  constexpr size_t index(size_t i) const {
    return i;
  }
};

/* TODO (far future) extra containers:
 * - padded_array
 * - padded_vector
 */

/*
 * user-facing container shortcuts
 */

template <char DIM, size_t N, typename T>
using array = dimension<DIM, T, array_impl<N>>;

template <char DIM, typename T>
using vector = dimension<DIM, T, vector_impl_unsized>;

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
  array<'x', 20, vector<'y', scalar<float>>> a;
  cout << "array tests:" << endl;
  cout << a % has_dim<'x'> << endl;
  cout << a % has_dim<'y'> << endl;
  cout << a % has_dim<'z'> << endl;
  cout << a % has_dims<'x'> << endl;
  cout << a % has_dims<'x', 'y'> << endl;
  cout << a % has_dims<'x', 'y', 'z'> << endl;

  auto as = a % resize<'y'>(30);

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
  cout << as % fixs<'x','y'>(3,5) % offset << endl;
  cout << as % idx<'x','y'>(3,5) << endl;
  cout << as % idx<'y','x'>(3,5) << endl;

  return 0;
}
