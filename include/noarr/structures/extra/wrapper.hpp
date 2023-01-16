#ifndef NOARR_STRUCTURES_WRAPPER_HPP
#define NOARR_STRUCTURES_WRAPPER_HPP

#include "../base/contain.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/funcs.hpp"
#include "../extra/to_struct.hpp"
#include "../structs/setters.hpp"
#include "../structs/slice.hpp"

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

template<class T>
struct is_cube<wrapper<T>> : is_cube<T> {};

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
		return wrap(base::template get<0>() ^ noarr::set_length<Dim>(length));
	}

	/**
	 * @brief fixes an index (or indices) given by dimension name(s) in the wrapped structure
	 * 
	 * @tparam Dims: the dimension names
	 * @param ts: parameters for fixing the indices
	 */
	template<char... Dims, class... Ts>
	constexpr auto fix(Ts... ts) const noexcept {
		return wrap(base::template get<0>() ^ noarr::fix<Dims...>(ts...));
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
		return wrap(base::template get<0>() ^ noarr::shift<Dims...>(ts...));
	}

	/**
	 * @brief returns an offset of a substructure
	 * 
	 * @tparam SubStruct: the substructure
	 */
	template<class SubStruct, char... Dims, class... Ts>
	constexpr auto offset(Ts... ts) const noexcept {
		return base::template get<0>() | noarr::offset<SubStruct, Dims...>(ts...);
	}

	/**
	 * @brief returns the number of indices in the structure specified by the dimension name
	 * 
	 * @tparam Dim: the dimension name of the desired structure
	 */
	template<char Dim, class... Ts>
	constexpr auto get_length(Ts... ts) const noexcept {
		return base::template get<0>() | noarr::get_length<Dim>(ts...);
	}

	/**
	 * @brief returns the size (in bytes) of the wrapped structure
	 */
	template<class... Ts>
	constexpr auto get_size(Ts... ts) const noexcept {
		return base::template get<0>() | noarr::get_size(ts...);
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
constexpr auto wrap() noexcept { return [](auto structure) constexpr noexcept {
	return wrap(structure);
}; }

template<class T>
struct to_struct<wrapper<T>> {
	using type = T;
	static constexpr T convert(wrapper<T> w) noexcept { return w.unwrap(); }
};

} // namespace noarr

#endif // NOARR_STRUCTURES_WRAPPER_HPP
