namespace tbb {

struct split {};

template<class R, class F>
inline void parallel_for(const R &range, const F &f) {
	range.is_divisible();
	f(range);
}

template<class T>
struct combinable {
	T t;
	T &local() { return t; }
	template<class F> void combine_each(const F &f) { f(std::as_const(t)); }
};

}
