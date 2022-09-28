#ifndef NOARR_STRUCTURES_VIEW_HPP
#define NOARR_STRUCTURES_VIEW_HPP

#include "struct_decls.hpp"
#include "contain.hpp"
#include "state.hpp"
#include "signature.hpp"

namespace noarr {

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
	static constexpr type convert(state<StateItem...> s) {
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
	constexpr auto instantiate_and_construct(Struct s) noexcept { return rename_t<Struct, DimPairs...>(s); }
};

template<char... DimPairs>
using rename = rename_proto<DimPairs...>;

template<char Dim, class T, class StartT>
struct shift_t : contain<T, StartT> {
	using base = contain<T, StartT>;
	using base::base;

	static constexpr char name[] = "shift_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<StartT>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr StartT start() const noexcept { return base::template get<1>(); }

	static_assert(T::signature::template all_accept<Dim>, "The structure does not have a dimension of this name");
private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		template<class L, class S>
		struct subtract { using type = dynamic_arg_length; };
		template<class S>
		struct subtract<unknown_arg_length, S> { using type = unknown_arg_length; };
		template<std::size_t L, std::size_t S>
		struct subtract<static_arg_length<L>, std::integral_constant<std::size_t, S>> { using type = static_arg_length<L-S>; };
		using type = function_sig<Dim, typename subtract<ArgLength, StartT>::type, RetSig>;
	};
	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(StartT::value || true, "Cannot shift a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t len = sizeof...(RetSigs) - start;

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;
		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> { using type = dep_function_sig<Dim, typename original::template ret_sig<Indices-start>...>; };

		using type = typename pack_helper<>::type;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		// TODO constexpr arithmetic
		auto tmp_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		if constexpr(State::template contains<index_in<Dim>>)
			if constexpr(State::template contains<length_in<Dim>>)
				return tmp_state.template with<index_in<Dim>, length_in<Dim>>(state.template get<index_in<Dim>>() + start(), state.template get<length_in<Dim>>() + start());
			else
				return tmp_state.template with<index_in<Dim>>(state.template get<index_in<Dim>>() + start());
		else
			if constexpr(State::template contains<length_in<Dim>>)
				return tmp_state.template with<length_in<Dim>>(state.template get<length_in<Dim>>() + start());
			else
				return tmp_state;
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
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			if constexpr(State::template contains<length_in<Dim>>) {
				// TODO check remaining state
				return state.template get<length_in<Dim>>();
			} else {
				return sub_structure().template length<Dim>(state.template remove<index_in<Dim>, length_in<Dim>>()) - start();
			}
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<char Dim, class StartT>
struct shift_proto : contain<StartT> {
	using base = contain<StartT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return shift_t<Dim, Struct, StartT>(s, base::template get<0>()); }
};

/**
 * @brief shifts an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dim: the dimension names
 * @param start: parameters for shifting the indices
 */
template<char... Dim, class... StartT>
constexpr auto shift(StartT... start) noexcept { return (neutral_proto() ^ ... ^ shift_proto<Dim, good_index_t<StartT>>(start)); }

template<char Dim, class T, class StartT, class LenT>
struct slice_t : contain<T, StartT, LenT> {
	using base = contain<T, StartT, LenT>;
	using base::base;

	static constexpr char name[] = "shift_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<StartT>,
		type_param<LenT>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr StartT start() const noexcept { return base::template get<1>(); }
	constexpr LenT len() const noexcept { return base::template get<2>(); }

	static_assert(T::signature::template all_accept<Dim>, "The structure does not have a dimension of this name");
private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> { using type = function_sig<Dim, arg_length_from_t<LenT>, RetSig>; };
	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(StartT::value || true, "Cannot slice a tuple dimension dynamically");
		static_assert(LenT::value || true, "Cannot slice a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t len = LenT::value;

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;
		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> { using type = dep_function_sig<Dim, typename original::template ret_sig<Indices-start>...>; };

		using type = typename pack_helper<>::type;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		// TODO constexpr arithmetic
		if constexpr(State::template contains<index_in<Dim>>)
			return state.template remove<index_in<Dim>>().template with<index_in<Dim>>(state.template get<index_in<Dim>>() + start());
		else
			return state;
	}

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		if constexpr(QDim == Dim) {
			static_assert(!State::template contains<index_in<Dim>>, "Index already set");
			// TODO check remaining state
			return len();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<char Dim, class StartT, class LenT>
struct slice_proto : contain<StartT, LenT> {
	using base = contain<StartT, LenT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return slice_t<Dim, Struct, StartT, LenT>(s, base::template get<0>(), base::template get<1>()); }
};

template<char Dim, class StartT, class LenT>
constexpr auto slice(StartT start, LenT len) noexcept { return slice_proto<Dim, good_index_t<StartT>, good_index_t<LenT>>(start, len); }

} // namespace noarr

#endif // NOARR_STRUCTURES_VIEW_HPP
