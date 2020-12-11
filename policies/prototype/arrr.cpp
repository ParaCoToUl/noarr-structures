#include <iostream>
#include <cassert>

using namespace std;
namespace arrr{

/* indexers */

template<typename ARR, char C, char... CS>
struct idx_t {
	template<typename... Args>
	constexpr size_t operator()(ARR a, size_t i, Args... args) {
		idx_t<ARR, C> curr;
		idx_t<ARR, CS...> next;
		return curr(a, i) + next(a, args...);
	};
};

template<typename ARR, char C>
struct idx_t<ARR,C> {
	constexpr size_t operator()(ARR a, size_t i) const {
		static_assert(a.template has_idx<C>(), "index not found");
		return a.template idx1<C>(i);
	}
};

/* data types */

template<char IDX, size_t N, typename T>
struct array {
	T t;
	constexpr size_t step() {return t.size();}
	constexpr size_t size() {return N*step();}

	template<char C>
	constexpr bool has_idx() {
		if constexpr (IDX==C) return true;
		return t.template has_idx<C>();
	}

	template<char C>
	constexpr size_t idx1(size_t i) {
		if constexpr (IDX==C) return i*step();
		else return t.template idx1<C>(i);
	}
	
	template<char... CS>
	constexpr idx_t<array<IDX,N,T>, CS...> idx(){
		return idx_t<array<IDX,N,T>, CS...>();
	}

  template<char C>
  void resize(size_t s) {
    static_assert(IDX!=C, "trying to resize an array");
    t.template resize<C>(s);
  }
};

template<char IDX, typename T>
struct vector{
  T t;
  size_t n;

  constexpr size_t step() {return t.size();}
  constexpr size_t size() {return n*step();}

	template<char C>
	constexpr bool has_idx() {
		if constexpr (IDX==C) return true;
		return t.template has_idx<C>();
	}

	template<char C>
	constexpr size_t idx1(size_t i) {
		if constexpr (IDX==C) return i*step();
		else return t.template idx1<C>(i);
	}
	
	template<char... CS>
	constexpr idx_t<vector<IDX,T>, CS...> idx(){
		return idx_t<vector<IDX,T>, CS...>();
	}

  template<char C>
  void resize(size_t s) {
		if constexpr (IDX==C) n=s;
    else t.template resize<C>(s);
  }
};

template<typename T>
struct scalar {
	static constexpr size_t step() {return sizeof(T);}
	static constexpr size_t size() {return sizeof(T);}

	template <char C>
	static constexpr bool has_idx() {
		return false;
	}
};

} //namespace arrr

using namespace arrr;

int main() {
	array<'x', 10, vector<'y', scalar<float>>> a;
	a.resize<'y'>(12300);

	std::cout << a.idx1<'x'>(3) << std::endl;
	std::cout << a.idx1<'y'>(3) << std::endl;

	auto idx = a.idx<'y','x'>();
	std::cout << idx(a, 1,2) << std::endl;
	return 0;
}