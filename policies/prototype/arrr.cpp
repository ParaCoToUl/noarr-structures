#include <iostream>
#include <cassert>

namespace arrr{

using std::size_t;

// really, someone wants to be explained what an array is?
template<size_t N, typename T, char IDX = '\0'>
struct array;

// vector: array with size given at runtime
template<typename T, char IDX = '\0'>
struct vector;

// vector which has been given a size
template<typename T, char IDX = '\0'>
struct sized_vector;

// for indexed structures already provided with an index
template<typename T>
struct offset;

// for ground types
template<typename T>
struct scalar;

// struct tuple;

// helpers:

template<typename T, T first, T... rest>
struct is_distinct;

template<typename T, typename, T first, T... rest>
struct unique_values_help { static constexpr bool value = false; };

template<typename T, T first, T... rest>
struct unique_values_help<T, typename std::enable_if<is_distinct<T, first, rest...>::value>::type, first, rest...> { static constexpr bool value = true; };

template<typename T, T first, T second, T... rest>
struct is_distinct<T, first, second, rest...> : is_distinct<T, first, rest...> {};

template<typename T, T first, T... rest>
struct is_distinct<T, first, first, rest...> { static constexpr bool value = false; };

template<typename T, T first, T second>
struct is_distinct<T, first, second> { static constexpr bool value = true; };

template<typename T, T first>
struct is_distinct<T, first, first> { static constexpr bool value = false; };

template<typename T, T first, T... rest>
using unique_values = unique_values_help<T, void, first, rest...>;

template<typename T>
struct is_scalar { static constexpr bool value = false; };

template<typename T>
struct is_scalar<scalar<T>> { static constexpr bool value = true; };

/* indexer */

template<typename S, char C, char... CS>
struct idx_t {
	static_assert(unique_values<char, C, CS...>::value, "indexes collide");
	static_assert(S::dynamic_index(), "structure does not provide dynamic indexing"); // TODO: dynamic or linear?
	using type = S;
	const S &s;
	template<typename Arg, typename... Args>
	constexpr size_t operator()(Arg arg, Args... args) const {
		static_assert(S::template has_idx<C>(), "index not found");
		static_assert(sizeof...(CS) == sizeof...(Args), "indexes given do not match indexes declared");
		if constexpr (S::IDX == C) {
			return s.get(arg) + s.template idx1<CS...>(arg)(args...);
		} else {
			return idx_t<S, CS..., C>{s}(args..., arg);
		}
	}
};

// TODO: add calling idx on idx_t?
// TODO: binding values for various indexes

template<typename S, char C>
struct idx_t<S, C> {
	static_assert(S::dynamic_index(), "structure does not provide dynamic indexing"); // TODO: same as above
	const S &s;
	template<typename... Args>
	constexpr size_t operator()(Args...) const {
		static_assert(1 == sizeof...(Args), "indexes given do not match indexes declared");
		return 0xDEAD; // FIXME
	}
	template<typename Arg>
	constexpr size_t operator()(Arg arg) const {
		static_assert(S::IDX == C, "index does not match");
		static_assert(is_scalar<typename S::item_t>::value, "path does not lead to ground item");
		return s.get(arg);
	}
};

/* data types */

template<size_t N, typename T, char I>
struct array  {
	T t;
	static constexpr char IDX = I;
	
	static constexpr bool dynamic() { return false; }
	static constexpr bool dynamic_index() { return true; }
	using item_t = T;

	constexpr size_t step() const { return t.size(); }
	constexpr size_t size() const { return N * step(); }

	template<size_t i>
	constexpr size_t get() const {
		return i * step();
	}

	constexpr size_t get(size_t i) const {
		return i * step();
	}

	template<size_t i, char... CS>
	constexpr idx_t<T, CS...> idx1() const {
		return idx_t<T, CS...>{t};
	}

	template<char... CS>
	constexpr idx_t<T, CS...> idx1(size_t i) const {
		return idx_t<T, CS...>{t};
	}

	template<char C>
	static constexpr bool has_idx() {
		if (C == IDX) return true;
		else return T::template has_idx<C>();
	}

	template<char... CS>
	constexpr idx_t<array, CS...> idx() const {
		return idx_t<array, CS...>{*this};
	}

	template<char C>
	void resize(size_t s) {
		static_assert(IDX != C, "trying to resize an array");
		t.template resize<C>(s);
	}
};

template<typename T, char I>
struct vector {
	T t;
	size_t n;
	static constexpr char IDX = I;
	
	static constexpr bool dynamic() { return true; }
	static constexpr bool dynamic_index() { return true; }
	using item_t = T;

	constexpr size_t step() const { return t.size(); }
	constexpr size_t size() const { /* TODO */ return n * step(); }

	template<size_t i>
	constexpr size_t get() const {
		return i * step();
	}

	constexpr size_t get(size_t i) const {
		return i * step();
	}

	template<size_t i, char... CS>
	constexpr idx_t<T, CS...> idx1() const {
		return idx_t<T, CS...>{t};
	}

	template<char... CS>
	constexpr idx_t<T, CS...> idx1(size_t i) const {
		return idx_t<T, CS...>{t};
	}
	
	template<char C>
	static constexpr bool has_idx() {
		if (C == IDX) return true;
		else return T::template has_idx<C>();
	}
	
	template<char... CS>
	constexpr idx_t<vector, CS...> idx() const {
		return idx_t<vector, CS...>{*this};
	}

	template<char C>
	void resize(size_t s) {
		if constexpr (IDX == C) n = s;
		else t.template resize<C>(s);
	}
};

template<typename T>
struct scalar {
	using item_t = T;

	static constexpr size_t step() { return sizeof(T); }
	static constexpr size_t size() { return sizeof(T); }

	template <char C>
	static constexpr bool has_idx() {
		return false;
	}

};

} //namespace arrr

using namespace arrr;

int main() {
	array<10, array<20,scalar<float>, 'y'>, 'x'> a;

	{
		auto idx = a.idx<'x', 'y'>();
		std::cout << idx(1, 2) << std::endl;
	}

	{
		auto idx = a.idx<'y', 'x'>();
		std::cout << idx(2, 1) << std::endl;
	}
	return 0;
}