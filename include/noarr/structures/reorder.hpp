#ifndef NOARR_STRUCTURES_REORDER_HPP
#define NOARR_STRUCTURES_REORDER_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "struct_decls.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "funcs.hpp"

namespace noarr {

namespace helpers {

template<char QDim, class State, class Signature>
struct reassemble_find;

template<char QDim, class State, class ArgLength, class RetSig>
struct reassemble_find<QDim, State, function_sig<QDim, ArgLength, RetSig>> {
	using type = function_sig<QDim, ArgLength, RetSig>;
};
template<char QDim, class State, class... RetSigs>
struct reassemble_find<QDim, State, dep_function_sig<QDim, RetSigs...>> {
	using type = dep_function_sig<QDim, RetSigs...>;
};
template<char QDim, class State, char Dim, class ArgLength, class RetSig>
struct reassemble_find<QDim, State, function_sig<Dim, ArgLength, RetSig>> {
	using type = typename reassemble_find<QDim, State, RetSig>::type;
};
template<char QDim, class State, char Dim, class... RetSigs>
struct reassemble_find<QDim, State, dep_function_sig<Dim, RetSigs...>> {
	static_assert(State::template contains<index_in<Dim>>, "Tuple indices must not be omitted or moved down");
	static constexpr std::size_t idx = State::template get_t<index_in<Dim>>::value;
	using ret_sig = typename dep_function_sig<Dim, RetSigs...>::ret_sig<idx>;
	using type = typename reassemble_find<QDim, State, ret_sig>::type;
};
template<char QDim, class State, class ValueType>
struct reassemble_find<QDim, State, scalar_sig<ValueType>> {
	static_assert(always_false_dim<QDim>, "The structure does not have a dimension of this name");
};

template<class T, class State>
struct reassemble_scalar;
template<char Dim, class ArgLength, class RetSig, class State>
struct reassemble_scalar<function_sig<Dim, ArgLength, RetSig>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<RetSig, State>::ty<>::pe; };
	template<class Useless>
	struct ty<false, Useless> { using pe = scalar_sig<void>; };
};
template<char Dim, class... RetSigs, class State>
struct reassemble_scalar<dep_function_sig<Dim, RetSigs...>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<typename dep_function_sig<Dim, RetSigs...>::ret_sig<State::template get_t<index_in<Dim>>::value>, State>::ty<>::pe; };
	template<class Useless>
	struct ty<false, Useless> { using pe = scalar_sig<void>; };
};
template<class ValueType, class State>
struct reassemble_scalar<scalar_sig<ValueType>, State> {
	template<class = void>
	struct ty { using pe = scalar_sig<ValueType>; };
};

template<class TopSig, class State, char... Dims>
struct reassemble_build;
template<class TopSig, class State, char Dim, char... Dims>
struct reassemble_build<TopSig, State, Dim, Dims...> {
	static_assert((... && (Dim != Dims)), "Duplicate dimension in reorder");
	using found = typename reassemble_find<Dim, State, TopSig>::type;
	template<class = found>
	struct ty;
	template<class ArgLength, class RetSig>
	struct ty<function_sig<Dim, ArgLength, RetSig>> {
		using sub_state = decltype(std::declval<State>().template with<index_in<Dim>>(std::size_t()));
		using sub_sig = typename reassemble_build<TopSig, sub_state, Dims...>::ty<>::pe;
		using pe = function_sig<Dim, ArgLength, sub_sig>;
	};
	template<class... RetSigs>
	struct ty<dep_function_sig<Dim, RetSigs...>> {
		template<std::size_t N>
		using sub_state = decltype(std::declval<State>().template with<index_in<Dim>>(std::integral_constant<std::size_t, N>()));
		template<std::size_t N>
		using sub_sig = typename reassemble_build<TopSig, sub_state<N>, Dims...>::ty<>::pe;

		template<class = std::index_sequence_for<RetSigs...>>
		struct pack_helper;
		template<std::size_t... N>
		struct pack_helper<std::index_sequence<N...>> { using type = dep_function_sig<Dim, sub_sig<N>...>; };

		using pe = typename pack_helper<>::type;
	};
	using type = typename reassemble_scalar<TopSig, State>::ty<>::pe;
};
template<class TopSig, class State>
struct reassemble_build<TopSig, State> : reassemble_scalar<TopSig, State> {};

template<class T>
struct reassemble_completeness;
template<char Dim, class ArgLength, class RetSig>
struct reassemble_completeness<function_sig<Dim, ArgLength, RetSig>> : reassemble_completeness<RetSig> {};
template<char Dim, class... RetSigs>
struct reassemble_completeness<dep_function_sig<Dim, RetSigs...>> : std::integral_constant<bool, (... && reassemble_completeness<RetSigs>::value)> {};
template<class ValueType>
struct reassemble_completeness<scalar_sig<ValueType>> : std::true_type {};
template<>
struct reassemble_completeness<scalar_sig<void>> : std::false_type {};

} // namespace helpers

template<class Signature, char... Dims>
using reassemble_sig = typename helpers::reassemble_build<Signature, state<>, Dims...>::ty<>::pe;

template<class ReassembledSignature>
static constexpr bool reassemble_is_complete = helpers::reassemble_completeness<ReassembledSignature>::value;

template<class T, char... Dims>
struct reorder_t : contain<T> {
	using base = contain<T>;
	using base::base;

	// TODO description

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	using signature = reassemble_sig<typename T::signature, Dims...>;
	static constexpr bool complete = reassemble_is_complete<signature>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(complete, "Some dimensions were omitted during reordering, cannot use the structure");
		return sub_structure().size(state);
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(complete, "Some dimensions were omitted during reordering, cannot use the structure");
		return offset_of<Sub>(sub_structure(), state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(complete, "Some dimensions were omitted during reordering, cannot use the structure");
		return sub_structure().template length<QDim>(state);
	}
};

template<char... Dims>
struct reorder_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return reorder_t<Struct, Dims...>(s); }
};

template<char... Dims>
using reorder = reorder_proto<Dims...>;

template<char Dim, class T>
struct hoist_t : contain<T> {
	using base = contain<T>;
	using base::base;

	// TODO description

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

private:
	using hoisted = typename helpers::reassemble_find<Dim, state<>, typename T::signature>::type; // reassemble_find also checks the dimension exists and is not within a tuple.
	static_assert(!hoisted::dependent, "Cannot hoist tuple dimension, use reorder");
	template<class Original>
	struct dim_replacement {
		static_assert(std::is_same_v<Original, hoisted>, "bug");
		using type = typename Original::ret_sig;
	};
public:
	using signature = function_sig<Dim, typename hoisted::arg_length, typename T::signature::replace<dim_replacement, Dim>>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		return sub_structure().size(state);
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), state);
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		return sub_structure().template length<QDim>(state);
	}
};

template<char Dim>
struct hoist_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return hoist_t<Dim, Struct>(s); }
};

template<char Dim>
using hoist = hoist_proto<Dim>;

} // namespace noarr

#endif // NOARR_STRUCTURES_REORDER_HPP
