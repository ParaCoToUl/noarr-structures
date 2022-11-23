#ifndef NOARR_STRUCTURES_VIEWS_HPP
#define NOARR_STRUCTURES_VIEWS_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

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
	static constexpr std::size_t idx = state_get_t<State, index_in<Dim>>::value;
	using ret_sig = typename dep_function_sig<Dim, RetSigs...>::template ret_sig<idx>;
	using type = typename reassemble_find<QDim, State, ret_sig>::type;
};
template<char QDim, class State, class ValueType>
struct reassemble_find<QDim, State, scalar_sig<ValueType>> {
	static_assert(value_always_false<QDim>, "The structure does not have a dimension of this name");
};

template<class T, class State>
struct reassemble_scalar;
template<char Dim, class ArgLength, class RetSig, class State>
struct reassemble_scalar<function_sig<Dim, ArgLength, RetSig>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<RetSig, State>::template ty<>::pe; };
	template<class Useless>
	struct ty<false, Useless> { using pe = scalar_sig<void>; };
};
template<char Dim, class... RetSigs, class State>
struct reassemble_scalar<dep_function_sig<Dim, RetSigs...>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<typename dep_function_sig<Dim, RetSigs...>::template ret_sig<state_get_t<State, index_in<Dim>>::value>, State>::template ty<>::pe; };
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
		using sub_sig = typename reassemble_build<TopSig, sub_state, Dims...>::template ty<>::pe;
		using pe = function_sig<Dim, ArgLength, sub_sig>;
	};
	template<class... RetSigs>
	struct ty<dep_function_sig<Dim, RetSigs...>> {
		template<std::size_t N>
		using sub_state = decltype(std::declval<State>().template with<index_in<Dim>>(std::integral_constant<std::size_t, N>()));
		template<std::size_t N>
		using sub_sig = typename reassemble_build<TopSig, sub_state<N>, Dims...>::template ty<>::pe;

		template<class = std::index_sequence_for<RetSigs...>>
		struct pack_helper;
		template<std::size_t... N>
		struct pack_helper<std::index_sequence<N...>> { using type = dep_function_sig<Dim, sub_sig<N>...>; };

		using pe = typename pack_helper<>::type;
	};
	using type = typename reassemble_scalar<TopSig, State>::template ty<>::pe;
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
using reassemble_sig = typename helpers::reassemble_build<Signature, state<>, Dims...>::template ty<>::pe;

template<class ReassembledSignature>
static constexpr bool reassemble_is_complete = helpers::reassemble_completeness<ReassembledSignature>::value;

template<class T, char... Dims>
struct reorder_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static constexpr char name[] = "reorder_t";
	using params = struct_params<
		structure_param<T>,
		dim_param<Dims>...>;

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
		static_assert(complete || signature::template any_accept<QDim>, "Some dimensions were omitted during reordering, cannot use the structure");
		return sub_structure().template length<QDim>(state);
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), state);
	}
};

template<char... Dims>
struct reorder_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return reorder_t<Struct, Dims...>(s); }
};

template<char... Dims>
using reorder = reorder_proto<Dims...>;

template<char Dim, class T>
struct hoist_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static constexpr char name[] = "hoist_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

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
	using signature = function_sig<Dim, typename hoisted::arg_length, typename T::signature::template replace<dim_replacement, Dim>>;

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

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), state);
	}
};

template<char Dim>
struct hoist_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return hoist_t<Dim, Struct>(s); }
};

template<char Dim>
using hoist = hoist_proto<Dim>;

namespace helpers {

template<class EvenAcc, class OddAcc, char... DimPairs>
struct rename_unzip_dim_pairs;
template<char... EvenAcc, char... OddAcc, char Even, char Odd, char... DimPairs>
struct rename_unzip_dim_pairs<std::integer_sequence<char, EvenAcc...>, std::integer_sequence<char, OddAcc...>, Even, Odd, DimPairs...>
	: rename_unzip_dim_pairs<std::integer_sequence<char, EvenAcc..., Even>, std::integer_sequence<char, OddAcc..., Odd>, DimPairs...> {};
template<class EvenAcc, class OddAcc>
struct rename_unzip_dim_pairs<EvenAcc, OddAcc> {
	using even = EvenAcc;
	using odd = OddAcc;
};

template<class>
struct rename_uniquity;
template<char Dim, char... Dims>
struct rename_uniquity<std::integer_sequence<char, Dim, Dims...>> {
	static constexpr bool value = (... && (Dims != Dim)) && rename_uniquity<std::integer_sequence<char, Dims...>>::value;
};
template<>
struct rename_uniquity<std::integer_sequence<char>> : std::true_type {};

template<char QDim, class From, class To>
struct rename_dim;
template<char QDim, char FromHead, char... FromTail, char ToHead, char... ToTail>
struct rename_dim<QDim, std::integer_sequence<char, FromHead, FromTail...>, std::integer_sequence<char, ToHead, ToTail...>>
	: rename_dim<QDim, std::integer_sequence<char, FromTail...>, std::integer_sequence<char, ToTail...>> {};
template<char FromHead, char... FromTail, char ToHead, char... ToTail>
struct rename_dim<FromHead, std::integer_sequence<char, FromHead, FromTail...>, std::integer_sequence<char, ToHead, ToTail...>> {
	static constexpr char dim = ToHead;
};
template<char QDim>
struct rename_dim<QDim, std::integer_sequence<char>, std::integer_sequence<char>> {
	static constexpr char dim = QDim;
};

template<class From, class To, class Signature>
struct rename_sig;
template<class From, class To, char Dim, class ArgLength, class RetSig>
struct rename_sig<From, To, function_sig<Dim, ArgLength, RetSig>> {
	using type = function_sig<rename_dim<Dim, From, To>::dim, ArgLength, typename rename_sig<From, To, RetSig>::type>;
};
template<class From, class To, char Dim, class... RetSigs>
struct rename_sig<From, To, dep_function_sig<Dim, RetSigs...>> {
	using type = dep_function_sig<rename_dim<Dim, From, To>::dim, typename rename_sig<From, To, RetSigs>::type...>;
};
template<class From, class To, class ValueType>
struct rename_sig<From, To, scalar_sig<ValueType>> {
	using type = scalar_sig<ValueType>;
};

template<class From, class To, class StateTag>
struct rename_state_tag;
template<class From, class To, char Dim>
struct rename_state_tag<From, To, index_in<Dim>> { using type = index_in<rename_dim<Dim, From, To>::dim>; };
template<class From, class To, char Dim>
struct rename_state_tag<From, To, length_in<Dim>> { using type = length_in<rename_dim<Dim, From, To>::dim>; };

template<class From, class To, class State>
struct rename_state;
template<class From, class To, class... StateItem>
struct rename_state<From, To, state<StateItem...>> {
	using type = state<state_item<typename rename_state_tag<From, To, typename StateItem::tag>::type, typename StateItem::value_type>...>;
	static constexpr type convert(state<StateItem...> s) noexcept {
		(void) s; // suppress warning about unused parameter when the pack below is empty
		return type(s.template get<typename StateItem::tag>()...);
	}
};

} // namespace helpers

template<class T, char... DimPairs>
struct rename_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static_assert(sizeof...(DimPairs) % 2 == 0, "Expected an even number of dimensions. Usage: rename<Old1, New1, Old2, New2, ...>()");
private:
	using unzip = helpers::rename_unzip_dim_pairs<std::integer_sequence<char>, std::integer_sequence<char>, DimPairs...>;
	using internal = typename unzip::even;
	using external = typename unzip::odd;
	template<class State>
	using rename_state = typename helpers::rename_state<external, internal, State>;
public:
	static_assert(helpers::rename_uniquity<internal>::value, "A dimension is renamed twice. Usage: rename<Old1, New1, Old2, New2, ...>()");
	static_assert(helpers::rename_uniquity<external>::value, "Multiple dimensions are renamed to the same name. Usage: rename<Old1, New1, Old2, New2, ...>()");

	static constexpr char name[] = "rename_t";
	using params = struct_params<
		structure_param<T>,
		dim_param<DimPairs>...>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

private:
	template<class = external, class = internal>
	struct assertion;
	template<char... ExternalDims, char... InternalDims>
	struct assertion<std::integer_sequence<char, ExternalDims...>, std::integer_sequence<char, InternalDims...>> {
		template<char Dim>
		static constexpr bool is_free = (!T::signature::template any_accept<Dim> || ... || (Dim == InternalDims)); // never used || used but renamed
		static_assert((... && T::signature::template any_accept<InternalDims>), "The structure does not have a dimension of a specified name. Usage: rename<Old1, New1, Old2, New2, ...>()");
		static_assert((... && is_free<ExternalDims>), "The structure already has a dimension of a specified name. Usage: rename<Old1, New1, Old2, New2, ...>()");
		// Note: in case a dimension is renamed to itself, is_free returns true. This is necessary to make the above assertion pass.
		// The `rename_uniquity<external>` check already ensures that if a dimension is renamed to itself, no other dimension is renamed to its name.
		static constexpr bool success = true;
	};
public:
	static_assert(assertion<>::success);
	using signature = typename helpers::rename_sig<internal, external, typename T::signature>::type;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		return rename_state<State>::convert(state);
	}

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		return sub_structure().size(sub_state<State>(state));
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state<State>(state));
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		return sub_structure().template length<helpers::rename_dim<QDim, external, internal>::dim>(sub_state<State>(state));
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state<State>(state));
	}
};

template<char... DimPairs>
struct rename_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return rename_t<Struct, DimPairs...>(s); }
};

template<char... DimPairs>
using rename = rename_proto<DimPairs...>;

} // namespace noarr

#endif // NOARR_STRUCTURES_VIEWS_HPP
