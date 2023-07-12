#ifndef NOARR_STRUCTURES_LAYOUTS_HPP
#define NOARR_STRUCTURES_LAYOUTS_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

/**
 * @brief tuple
 * 
 * @tparam Dim dimension added by the structure
 * @tparam T,TS... substructure types
 */
template<IsDim auto Dim, class... TS>
struct tuple_t : contain<TS...> {
	using base = contain<TS...>;

	template<std::size_t Index>
	constexpr auto sub_structure() const noexcept { return base::template get<Index>(); }
	static constexpr char name[] = "tuple_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<TS>...>;

	constexpr tuple_t() noexcept = default;
	explicit constexpr tuple_t(TS... ss) noexcept requires (sizeof...(TS) > 0) : base(ss...) {}

	static_assert(!(TS::signature::template any_accept<Dim> || ...), "Dimension name already used");
	using signature = dep_function_sig<Dim, typename TS::signature...>;

	constexpr auto size(IsState auto state) const noexcept {
		static_assert(!decltype(state)::template contains<length_in<Dim>>, "Cannot set tuple length");
		return size_inner(is, state.template remove<index_in<Dim>>());
	}

	template<class Sub>
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!decltype(state)::template contains<length_in<Dim>>, "Cannot set tuple length");
		static_assert(decltype(state)::template contains<index_in<Dim>>, "All indices must be set");
		static_assert(state_get_t<decltype(state), index_in<Dim>>::value || true, "Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
		constexpr std::size_t index = state_get_t<decltype(state), index_in<Dim>>::value;
		auto sub_state = state.template remove<index_in<Dim>>();
		return size_inner(std::make_index_sequence<index>(), sub_state) + offset_of<Sub>(sub_structure<index>(), sub_state);
	}

	template<IsDim auto QDim>
	constexpr auto length(IsState auto state) const noexcept {
		static_assert(!decltype(state)::template contains<length_in<Dim>>, "Cannot set tuple length");
		if constexpr(QDim == Dim) {
			static_assert(!decltype(state)::template contains<index_in<Dim>>, "Index already set");
			return constexpr_arithmetic::make_const<sizeof...(TS)>();
		} else {
			static_assert(decltype(state)::template contains<index_in<Dim>>, "Tuple indices must be set");
			constexpr std::size_t index = state_get_t<decltype(state), index_in<Dim>>::value;
			return sub_structure<index>().template length<QDim>(state.template remove<index_in<Dim>>());
		}
	}

	template<class Sub>
	constexpr void strict_state_at(IsState auto) const noexcept {
		static_assert(value_always_false<Dim>, "A tuple cannot be used in this context");
	}

private:
	static constexpr std::index_sequence_for<TS...> is = {};

	template<std::size_t... IS>
	constexpr auto size_inner(std::index_sequence<IS...>, IsState auto sub_state) const noexcept {
		using namespace constexpr_arithmetic;
		(void) sub_state; // don't complain about unused parameter in case of empty fold
		return (make_const<0>() + ... + sub_structure<IS>().size(sub_state));
	}
};

template<IsDim auto Dim>
struct tuple_proto {
	static constexpr bool proto_preserves_layout = false;

	template<class... Structs>
	constexpr auto instantiate_and_construct(Structs... s) const noexcept { return tuple_t<Dim, Structs...>(s...); }
};

template<IsDim auto Dim>
constexpr auto tuple() noexcept {
	return tuple_proto<Dim>();
}

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 * 
 * @tparam Dim: the dimmension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<IsDim auto Dim, class T>
struct vector_t : contain<T> {
	static constexpr char name[] = "vector_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	constexpr vector_t() noexcept = default;
	explicit constexpr vector_t(T sub_structure) noexcept : contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return contain<T>::template get<0>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, unknown_arg_length, typename T::signature>;

	constexpr auto size(IsState auto state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(decltype(state)::template contains<length_in<Dim>>, "Unknown vector length");
		auto len = state.template get<length_in<Dim>>();
		return len * sub_structure().size(state.template remove<index_in<Dim>, length_in<Dim>>());
	}

	template<class Sub>
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(decltype(state)::template contains<index_in<Dim>>, "All indices must be set");
		if constexpr(!std::is_same_v<decltype(state.template get<length_in<Dim>>()), std::integral_constant<std::size_t, 1>>) {
			// offset = index * elem_size + offset_within_elem
			auto index = state.template get<index_in<Dim>>();
			auto sub_struct = sub_structure();
			auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
			return index * sub_struct.size(sub_state) + offset_of<Sub>(sub_struct, sub_state);
		} else {
			// Optimization: length is one, thus the only valid index is zero.
			// Assume the index is valid (caller's responsibility).
			// offset = 0 * elem_size + offset_within_elem = offset_within_elem
			return offset_of<Sub>(sub_structure(), state.template remove<index_in<Dim>, length_in<Dim>>());
		}
	}

	template<IsDim auto QDim>
	constexpr auto length(IsState auto state) const noexcept {
		if constexpr(QDim == Dim) {
			static_assert(!decltype(state)::template contains<index_in<Dim>>, "Index already set");
			static_assert(decltype(state)::template contains<length_in<Dim>>, "This length has not been set yet");
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(state.template remove<index_in<Dim>, length_in<Dim>>());
		}
	}

	template<class Sub>
	constexpr void strict_state_at(IsState auto) const noexcept {
		static_assert(value_always_false<Dim>, "A vector cannot be used in this context");
	}
};

template<IsDim auto Dim>
struct vector_proto {
	static constexpr bool proto_preserves_layout = false;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return vector_t<Dim, Struct>(s); }
};

template<IsDim auto Dim>
constexpr auto vector() noexcept {
	return vector_proto<Dim>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_LAYOUTS_HPP
