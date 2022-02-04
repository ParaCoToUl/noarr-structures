#ifndef NOARR_STRUCTURES_WRAPPER_HPP
#define NOARR_STRUCTURES_WRAPPER_HPP

#include "funcs.hpp"

namespace noarr {

/**
 * @brief wraps the structure and offers the `.` notation instead of the `|` notation in applying functions to a structure (structure always on the left side)
 * 
 * @tparam Structure: the underlying structure
 */
template<class Structure>
class wrapper;

/**
 * @brief wraps the structure into a `wrapper`
 * 
 * @param s: the structure to be wrapped
 */
template<class Structure>
constexpr wrapper<Structure> wrap(Structure s) noexcept;

namespace helpers {

template<class T>
struct is_cube_impl<wrapper<T>> : is_cube<T> {};

struct wrap_impl {
	using func_family = top_tag;

	template<class Structure>
	constexpr auto operator()(Structure structure) const noexcept {
		return wrap(structure);
	}
};

}

template<class Structure>
class wrapper : private contain<Structure> {
	using base = contain<Structure>;

public:
	constexpr wrapper() noexcept = default;
	explicit constexpr wrapper(Structure s) noexcept : base(s) {}

	/**
	 * @brief sets the length of a `vector`, `sized_vector` or an `array` in the wrapped structure
	 * 
	 * @tparam Dim: the dimension name of the transfromed structure
	 * @param length: the desired length
	 */
	template<char Dim>
	constexpr auto set_length(std::size_t length) const noexcept {
		return wrap(base::template get<0>() | noarr::set_length<Dim>(length));
	}

	/**
	 * @brief fixes an index (or indices) given by dimension name(s) in the wrapped structure
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: parameters for fixing the indices
	 */
	template<char... Dims, class... Ts>
	constexpr auto fix(Ts... ts) const noexcept {
		return wrap(base::template get<0>() | noarr::fix<Dims...>(ts...));
	}

	/**
	 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item
	 * 
	 * @tparam Dims: the dimension names of fixed indices
	 * @param ts: parameters for fixing the indices
	 */
	template<char... Dims, class... Ts>
	constexpr auto offset(Ts... ts) const noexcept {
		return base::template get<0>() | noarr::offset<Dims...>(ts...);
	}

	/**
	 * @brief optionally fixes indices (see `fix`) and then returns the offset of the resulting item
	 * 
	 * @tparam Dims: the dimension names of fixed indices
	 * @param ts: parameters for fixing the indices
	 */
	template<char... Dims, class... Ts>
	constexpr auto shift(Ts... ts) const noexcept {
		return wrap(base::template get<0>() | noarr::shift<Dims...>(ts...));
	}

	/**
	 * @brief returns an offset of a substructure with a certain index in a structure given by its dimension name
	 * 
	 * @tparam Dim: the dimension name
	 * @param t: the index of the substructure
	 */
	template<char Dim, class T>
	constexpr auto get_offset(T t) const noexcept {
		return base::template get<0>() | noarr::get_offset<Dim>(t);
	}

	/**
	 * @brief returns the number of indices in the structure specified by the dimension name
	 * 
	 * @tparam Dim: the dimension name of the desired structure
	 */
	template<char Dim>
	constexpr auto get_length() const noexcept {
		return base::template get<0>() | noarr::get_length<Dim>();
	}

	/**
	 * @brief returns the size (in bytes) of the wrapped structure
	 */
	constexpr auto get_size() const noexcept {
		return base::template get<0>() | noarr::get_size();
	}

	/**
	 * @brief returns the item in the blob specified by `ptr` offset of which is specified by the wrapped structure structure, optionally, with some fixed indices (see `fix`)
	 * @tparam Dims: the dimension names of the fixed dimensions
	 * @param ptr: the pointer to blob structure
	 */
	template<char... Dims, class V, class... Ts>
	constexpr decltype(auto) get_at(V *ptr, Ts... ts) const noexcept {
		return base::template get<0>() | noarr::get_at<Dims...>(ptr, ts...);
	}

	/**
	 * @brief returns the wrapped structure
	 */
	constexpr auto unwrap() const noexcept {
		return base::template get<0>();
	}
};

template<class Structure>
constexpr wrapper<Structure> wrap(Structure s) noexcept {
	return wrapper<Structure>(s);
}

/**
 * @brief wraps a structure into a `wrapper`
 * 
 */
constexpr auto wrap() noexcept {
	return helpers::wrap_impl();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_WRAPPER_HPP
