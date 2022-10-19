#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include <type_traits>
#include <utility>

#include "state.hpp"
#include "structs_common.hpp"
#include "struct_traits.hpp"
#include "scalar.hpp"

namespace noarr {

template<std::size_t I>
struct idx_t : std::integral_constant<std::size_t, I> {
    auto operator()() = delete; // using `idx<42>()` by mistake should be rejected, not evaluate to dynamic size_t of 42
};

template<std::size_t I>
constexpr idx_t<I> idx;

template<char Dim, class State>
constexpr auto get_length(State state) noexcept { return [state](auto structure) constexpr noexcept {
	return structure.template length<Dim>(state);
}; }

/**
 * @brief returns the number of indices in the structure specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the desired structure
 */
template<char Dim>
constexpr auto get_length() noexcept { return get_length<Dim, state<>>(empty_state); }

template<class SubStruct, class State>
constexpr auto offset(State state) noexcept { return [state](auto structure) constexpr noexcept {
	return offset_of<SubStruct>(structure, state);
}; }

template<class SubStruct, char... Dims, class... Idxs>
constexpr auto offset(Idxs... idxs) noexcept { return offset<SubStruct>(empty_state.with<index_in<Dims>...>(idxs...)); }

template<class State>
constexpr auto offset(State state) noexcept { return [state](auto structure) constexpr noexcept {
	using type = scalar_t<decltype(structure), State>;
	return offset_of<scalar<type>>(structure, state);
}; }

template<char... Dims, class... Idxs>
constexpr auto offset(Idxs... idxs) noexcept { return offset(empty_state.with<index_in<Dims>...>(idxs...)); }

template<class State>
constexpr auto get_size(State state) noexcept { return [state](auto structure) constexpr noexcept {
	return structure.size(state);
}; }

/**
 * @brief returns the size (in bytes) of the structure
 */
constexpr auto get_size() noexcept { return get_size(empty_state); }

namespace helpers {

template<class T>
constexpr auto sub_ptr(void *ptr, std::size_t off) noexcept { return (T*) ((char*) ptr + off); }
template<class T>
constexpr auto sub_ptr(const void *ptr, std::size_t off) noexcept { return (const T*) ((const char*) ptr + off); }
template<class T>
constexpr auto sub_ptr(volatile void *ptr, std::size_t off) noexcept { return (volatile T*) ((volatile char*) ptr + off); }
template<class T>
constexpr auto sub_ptr(const volatile void *ptr, std::size_t off) noexcept { return (const volatile T*) ((const volatile char*) ptr + off); }

} // namespace helpers

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 * 
 * @param ptr: the pointer to blob structure
 */
template<class State, class CvVoid>
constexpr auto get_at(CvVoid *ptr, State state) noexcept { return [ptr, state](auto structure) constexpr noexcept -> decltype(auto) {
	using type = scalar_t<decltype(structure), State>;
	return *helpers::sub_ptr<type>(ptr, offset_of<scalar<type>>(structure, state));
}; }

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, class... Idxs, class CvVoid>
constexpr auto get_at(CvVoid *ptr, Idxs... idxs) noexcept { return get_at(ptr, empty_state.with<index_in<Dims>...>(idxs...)); }

/**
 * @brief performs a simple application of `F` to `S`
 * 
 * @tparam S: the structure type
 * @tparam F: the function type
 * @param s: the structure
 * @param f: the function
 * @return the result of the piping
 */
template<class S, class F>
constexpr auto operator|(S s, F f) noexcept ->
std::enable_if_t<is_struct<std::enable_if_t<std::is_class<S>::value, S>>::value, decltype(std::declval<F>()(std::declval<S>()))> {
	return f(s);
}

} // namespace noarr

#endif // NOARR_STRUCTURES_FUNCS_HPP
