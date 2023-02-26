#ifndef NOARR_STRUCTURES_FUNCS_HPP
#define NOARR_STRUCTURES_FUNCS_HPP

#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/struct_traits.hpp"
#include "../extra/to_struct.hpp"
#include "../structs/scalar.hpp"

namespace noarr {

template<std::size_t I>
struct lit_t : std::integral_constant<std::size_t, I> {
	auto operator()() = delete; // using `lit<42>()` by mistake should be rejected, not evaluate to dynamic size_t of 42
};

template<std::size_t I>
constexpr lit_t<I> lit;

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

template<class T, class F>
struct sub_ptr_allowed { static constexpr bool value = false; };
template<class T>
struct sub_ptr_allowed<T, T> { static constexpr bool value = true; };
template<class T>
struct sub_ptr_allowed<T, void> { static constexpr bool value = true; };
template<class T>
struct sub_ptr_allowed<T, char> { static constexpr bool value = true; };
template<class T>
struct sub_ptr_allowed<T, unsigned char> { static constexpr bool value = true; };

template<>
struct sub_ptr_allowed<char, char> { static constexpr bool value = true; };
template<>
struct sub_ptr_allowed<unsigned char, unsigned char> { static constexpr bool value = true; };

template<class T, class F>
constexpr bool sub_ptr_allowed_v = sub_ptr_allowed<T, F>::value;

template<class T, class F>
constexpr auto sub_ptr(F *ptr, std::size_t off) noexcept
	-> std::enable_if_t<sub_ptr_allowed_v<T, F>, T*> { return (T*) ((char*) ptr + off); }
template<class T, class F>
constexpr auto sub_ptr(const F *ptr, std::size_t off) noexcept
	-> std::enable_if_t<sub_ptr_allowed_v<T, F>, const T*> { return (const T*) ((const char*) ptr + off); }
template<class T, class F>
constexpr auto sub_ptr(volatile F *ptr, std::size_t off) noexcept
	-> std::enable_if_t<sub_ptr_allowed_v<T, F>, volatile T*> { return (volatile T*) ((volatile char*) ptr + off); }
template<class T, class F>
constexpr auto sub_ptr(const volatile F *ptr, std::size_t off) noexcept
	-> std::enable_if_t<sub_ptr_allowed_v<T, F>, const volatile T*> { return (const volatile T*) ((const volatile char*) ptr + off); }

} // namespace helpers

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure
 *
 * @param ptr: the pointer to blob structure
 */
template<class State, class CvPtr>
constexpr auto get_at(CvPtr ptr, State state) noexcept { return [ptr, state](auto structure) constexpr noexcept -> decltype(auto) {
	using type = scalar_t<decltype(structure), State>;
	return *helpers::sub_ptr<type, std::remove_cv_t<std::remove_pointer_t<CvPtr>>>(ptr, offset_of<scalar<type>>(structure, state));
}; }

/**
 * @brief returns the item in the blob specified by `ptr` offset of which is specified by a structure with some fixed indices (see `fix`)
 * @tparam Dims: the dimension names of the fixed dimensions
 * @param ptr: the pointer to blob structure
 */
template<char... Dims, class... Idxs, class CvPtr>
constexpr auto get_at(CvPtr ptr, Idxs... idxs) noexcept { return get_at(ptr, empty_state.with<index_in<Dims>...>(idxs...)); }

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
constexpr auto operator|(const S &s, F f) noexcept -> decltype(std::declval<F>()(std::declval<typename to_struct<S>::type>())) {
	return f(to_struct<S>::convert(s));
}

} // namespace noarr

#endif // NOARR_STRUCTURES_FUNCS_HPP
