#ifndef NOARR_STRUCTURES_SETTERS_HPP
#define NOARR_STRUCTURES_SETTERS_HPP

#include "struct_decls.hpp"
#include "contain.hpp"
#include "state.hpp"
#include "type.hpp"

namespace noarr {


template<char Dim, class T, class IdxT>
struct fix_t : contain<T, IdxT> {
	using base = contain<T, IdxT>;
	using base::base;

	// TODO description

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr IdxT idx() const noexcept { return base::template get<1>(); }

	static_assert(T::struct_type::template all_accept<Dim>, "The structure does not have a dimension of this name");
private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetType>
	struct dim_replacement<function_type<Dim, ArgLength, RetType>> {
		static_assert(ArgLength::is_known, "Index cannot be fixed until its length is set");
		using type = RetType;
	};
	template<class... RetTypes>
	struct dim_replacement<dep_function_type<Dim, RetTypes...>> {
		using original = dep_function_type<Dim, RetTypes...>;
		static_assert(IdxT::value || true, "Tuple index must be set statically, add _idx to the index (e.g. replace 42 with 42_idx)");
		using type = typename original::ret_type<IdxT::value>;
	};
public:
	using struct_type = typename T::struct_type::replace<dim_replacement, Dim>;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		static_assert(!State::template contains<index_in<Dim>>, "This dimension is already fixed, it cannot be used from outside");
		static_assert(!State::template contains<length_in<Dim>>, "This dimension is already fixed, it cannot be used from outside");
		return state.template with<index_in<Dim>>(idx());
	}

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(QDim != Dim, "This dimension is already fixed, it cannot be used from outside");
		return sub_structure().template length<QDim>(sub_state(state));
	}
};

template<char Dim, class IdxT>
struct fix_proto : contain<IdxT> {
	using base = contain<IdxT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return fix_t<Dim, Struct, IdxT>(s, base::template get<0>()); }
};

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<char... Dim, class... IdxT>
constexpr auto fix(IdxT... idx) noexcept { return (unit_struct ^ ... ^ fix_proto<Dim, good_index_t<IdxT>>(idx)); }
template<char... Dim, class... IdxT>
constexpr auto fix(state<state_item<index_in<Dim>, IdxT>...> state) noexcept { return (unit_struct ^ ... ^ fix_proto<Dim, IdxT>(state.template get<index_in<Dim>>)); }
template<class... StateItem>
constexpr void fix(state<StateItem...>) {
	static_assert(always_false<state<StateItem...>>, "Unrecognized items in state. The fix(state) variant can be only used to fix indices (index_in<...>)");
}

template<char Dim, class T, class LenT>
struct set_length_t : contain<T, LenT> {
	using base = contain<T, LenT>;
	using base::base;

	// TODO description

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr LenT len() const noexcept { return base::template get<1>(); }

	static_assert(T::struct_type::template all_accept<Dim>, "The structure does not have a dimension of this name");
private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetType>
	struct dim_replacement<function_type<Dim, ArgLength, RetType>> {
		static_assert(!ArgLength::is_known, "The length in this dimension is already set");
		using type = function_type<Dim, arg_length_from_t<LenT>, RetType>;
	};
	template<class... RetTypes>
	struct dim_replacement<dep_function_type<Dim, RetTypes...>> {
		static_assert(always_false<Dim>, "Cannot set tuple length");
	};
public:
	using struct_type = typename T::struct_type::replace<dim_replacement, Dim>;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "The length in this dimension is already set");
		return state.template with<length_in<Dim>>(len());
	}

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		return sub_structure().template length<QDim>(sub_state(state));
	}
};

template<char Dim, class LenT>
struct set_length_proto : contain<LenT> {
	using base = contain<LenT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return set_length_t<Dim, Struct, LenT>(s, base::template get<0>()); }
};

/**
 * @brief sets the length of a `vector` specified by the dimension name
 * 
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<char... Dim, class... LenT>
constexpr auto set_length(LenT... len) noexcept { return (unit_struct ^ ... ^ set_length_proto<Dim, good_index_t<LenT>>(len)); }

} // namespace noarr

#endif // NOARR_STRUCTURES_SETTERS_HPP
