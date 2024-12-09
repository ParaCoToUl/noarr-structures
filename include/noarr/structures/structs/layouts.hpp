#ifndef NOARR_STRUCTURES_LAYOUTS_HPP
#define NOARR_STRUCTURES_LAYOUTS_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

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
 * @tparam T,TS ...substructure types
 */
template<IsDim auto Dim, class ...TS>
struct tuple_t : strict_contain<TS...> {
	using base = strict_contain<TS...>;
	static constexpr char name[] = "tuple_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<TS>...>;

	template<std::size_t Index>
	constexpr auto sub_structure() const noexcept { return this->template get<Index>(); }
	constexpr auto sub_state(IsState auto state) const noexcept { return state.template remove<index_in<Dim>>(); }

	constexpr tuple_t() noexcept = default;
	explicit constexpr tuple_t(TS ...ss) noexcept requires (sizeof...(TS) > 0) : base(ss...) {}

	static_assert(!(... || TS::signature::template any_accept<Dim>), "Dimension name already used");
	using signature = dep_function_sig<Dim, typename TS::signature...>;

	template<IsState State>
	constexpr auto size(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		return size_inner(is, sub_state(state));
	}

	template<class Sub, IsState State>
	constexpr auto strict_offset_of(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		static_assert(state_get_t<State, index_in<Dim>>::value || true, "Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
		constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;
		const auto sub_stat = sub_state(state);
		return size_inner(std::make_index_sequence<index>(), sub_stat) + offset_of<Sub>(sub_structure<index>(), sub_stat);
	}

	template<auto QDim, IsState State> requires (QDim != Dim || HasNotSetIndex<State, QDim>) && IsDim<decltype(QDim)>
	constexpr auto length(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set tuple length");
		if constexpr(QDim == Dim) {
			return constexpr_arithmetic::make_const<sizeof...(TS)>();
		} else {
			static_assert(State::template contains<index_in<Dim>>, "Tuple indices must be set");
			constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;
			return sub_structure<index>().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub>
	constexpr void strict_state_at(IsState auto) const noexcept {
		static_assert(value_always_false<Dim>, "A tuple cannot be used in this context");
	}

private:
	static constexpr std::index_sequence_for<TS...> is = {};

	template<std::size_t ...IS>
	constexpr auto size_inner(std::index_sequence<IS...>, [[maybe_unused]] IsState auto sub_state) const noexcept {
		using namespace constexpr_arithmetic;
		return (make_const<0>() + ... + sub_structure<IS>().size(sub_state));
	}
};

template<IsDim auto Dim>
struct tuple_proto {
	static constexpr bool proto_preserves_layout = false;

	template<class ...Structs>
	constexpr auto instantiate_and_construct(Structs ...s) const noexcept { return tuple_t<Dim, Structs...>(s...); }
};

template<IsDim auto Dim>
constexpr auto tuple() noexcept {
	return tuple_proto<Dim>();
}

/**
 * @brief unsized vector ready to be resized to the desired size, this vector does not have size yet
 *
 * @tparam Dim: the dimension name added by the vector
 * @tparam T: type of the substructure the vector contains
 */
template<IsDim auto Dim, class T>
struct vector_t : strict_contain<T> {
	static constexpr char name[] = "vector_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	constexpr vector_t() noexcept = default;
	explicit constexpr vector_t(T sub_structure) noexcept : strict_contain<T>(sub_structure) {}

	constexpr T sub_structure() const noexcept { return strict_contain<T>::get(); }
	constexpr auto sub_state(IsState auto state) const noexcept { return state.template remove<index_in<Dim>, length_in<Dim>>(); }

	static_assert(!T::signature::template any_accept<Dim>, "Dimension name already used");
	using signature = function_sig<Dim, unknown_arg_length, typename T::signature>;

	template<IsState State>
	constexpr auto size(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(State::template contains<length_in<Dim>>, "Unknown vector length");
		const auto len = state.template get<length_in<Dim>>();
		return len * sub_structure().size(sub_state(state));
	}

	template<class Sub, IsState State>
	constexpr auto strict_offset_of(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(State::template contains<index_in<Dim>>, "All indices must be set");
		if constexpr(!std::is_same_v<decltype(state.template get<length_in<Dim>>()), std::integral_constant<std::size_t, 1>>) {
			// offset = index * elem_size + offset_within_elem
			const auto index = state.template get<index_in<Dim>>();
			const auto sub_struct = sub_structure();
			const auto sub_stat = sub_state(state);
			return index * sub_struct.size(sub_stat) + offset_of<Sub>(sub_struct, sub_stat);
		} else {
			// Optimization: length is one, thus the only valid index is zero.
			// Assume the index is valid (caller's responsibility).
			// offset = 0 * elem_size + offset_within_elem = offset_within_elem
			return offset_of<Sub>(sub_structure(), sub_state(state));
		}
	}

	template<auto QDim, IsState State> requires (QDim != Dim || HasNotSetIndex<State, QDim>) && IsDim<decltype(QDim)>
	constexpr auto length(State state) const noexcept {
		if constexpr(QDim == Dim) {
			static_assert(State::template contains<length_in<Dim>>, "This length has not been set yet");
			return state.template get<length_in<Dim>>();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	constexpr void strict_state_at(State) const noexcept {
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
