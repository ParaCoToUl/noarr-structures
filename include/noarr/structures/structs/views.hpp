#ifndef NOARR_STRUCTURES_VIEWS_HPP
#define NOARR_STRUCTURES_VIEWS_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"
#include "../extra/sig_utils.hpp"

namespace noarr {

namespace helpers {

template<class T, IsState State>
struct reassemble_scalar;
template<IsDim auto Dim, class ArgLength, class RetSig, IsState State>
struct reassemble_scalar<function_sig<Dim, ArgLength, RetSig>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<RetSig, State>::template ty<>::pe; };
	template<class Useless>
	struct ty<false, Useless> { using pe = scalar_sig<void>; };
};
template<IsDim auto Dim, class ...RetSigs, IsState State>
struct reassemble_scalar<dep_function_sig<Dim, RetSigs...>, State> {
	template<bool cond = State::template contains<index_in<Dim>>, class = void>
	struct ty;
	template<class Useless>
	struct ty<true, Useless> { using pe = typename reassemble_scalar<typename dep_function_sig<Dim, RetSigs...>::template ret_sig<state_get_t<State, index_in<Dim>>::value>, State>::template ty<>::pe; };
	template<class Useless>
	struct ty<false, Useless> { using pe = scalar_sig<void>; };
};
template<class ValueType, IsState State>
struct reassemble_scalar<scalar_sig<ValueType>, State> {
	template<class = void>
	struct ty { using pe = scalar_sig<ValueType>; };
};

template<class TopSig, IsState State, auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct reassemble_build;
template<class TopSig, IsState State, IsDim auto Dim, auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct reassemble_build<TopSig, State, Dim, Dims...> {
	static_assert((... && (Dim != Dims)), "Duplicate dimension in reorder");
	using found = sig_find_dim<Dim, State, TopSig>;
	template<class = found>
	struct ty;
	template<class ArgLength, class RetSig>
	struct ty<function_sig<Dim, ArgLength, RetSig>> {
		using sub_state = decltype(std::declval<State>().template with<index_in<Dim>>(std::size_t()));
		using sub_sig = typename reassemble_build<TopSig, sub_state, Dims...>::template ty<>::pe;
		using pe = function_sig<Dim, ArgLength, sub_sig>;
	};
	template<class ...RetSigs>
	struct ty<dep_function_sig<Dim, RetSigs...>> {
		template<std::size_t N>
		using sub_state = decltype(std::declval<State>().template with<index_in<Dim>>(std::integral_constant<std::size_t, N>()));
		template<std::size_t N>
		using sub_sig = typename reassemble_build<TopSig, sub_state<N>, Dims...>::template ty<>::pe;

		template<class = std::index_sequence_for<RetSigs...>>
		struct pack_helper;
		template<std::size_t ...N>
		struct pack_helper<std::index_sequence<N...>> { using type = dep_function_sig<Dim, sub_sig<N>...>; };

		using pe = typename pack_helper<>::type;
	};
	using type = typename reassemble_scalar<TopSig, State>::template ty<>::pe;
};
template<class TopSig, IsState State>
struct reassemble_build<TopSig, State> : reassemble_scalar<TopSig, State> {};

template<class T>
struct reassemble_completeness;
template<IsDim auto Dim, class ArgLength, class RetSig>
struct reassemble_completeness<function_sig<Dim, ArgLength, RetSig>> : reassemble_completeness<RetSig> {};
template<IsDim auto Dim, class ...RetSigs>
struct reassemble_completeness<dep_function_sig<Dim, RetSigs...>> : std::integral_constant<bool, (... && reassemble_completeness<RetSigs>::value)> {};
template<class ValueType>
struct reassemble_completeness<scalar_sig<ValueType>> : std::true_type {};
template<>
struct reassemble_completeness<scalar_sig<void>> : std::false_type {};

} // namespace helpers

template<class Signature, auto ...Dims> requires IsDimPack<decltype(Dims)...>
using reassemble_sig = typename helpers::reassemble_build<Signature, state<>, Dims...>::template ty<>::pe;

template<class ReassembledSignature>
static constexpr bool reassemble_is_complete = helpers::reassemble_completeness<ReassembledSignature>::value;

template<class T, auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct reorder_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "reorder_t";
	using params = struct_params<
		structure_param<T>,
		dim_param<Dims>...>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

	using signature = reassemble_sig<typename T::signature, Dims...>;
	static constexpr bool complete = reassemble_is_complete<signature>;

	[[nodiscard]]
	constexpr auto size(IsState auto state) const noexcept {
		static_assert(complete, "Some dimensions were omitted during reordering, cannot use the structure");
		return sub_structure().size(state);
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		static_assert(complete, "Some dimensions were omitted during reordering, cannot use the structure");
		return offset_of<Sub>(sub_structure(), state);
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		static_assert(complete || signature::template any_accept<QDim>, "Some dimensions were omitted during reordering, cannot use the structure");
		return sub_structure().template length<QDim>(state);
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), state);
	}
};

template<auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct reorder_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return reorder_t<Struct, Dims...>(s); }
};

template<auto ...Dims> requires IsDimPack<decltype(Dims)...>
using reorder = reorder_proto<Dims...>;

template<IsDim auto Dim, class T>
struct hoist_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "hoist_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

private:
	using hoisted = sig_find_dim<Dim, state<>, typename T::signature>; // sig_find_dim also checks the dimension exists and is not within a tuple.
	static_assert(!hoisted::dependent, "Cannot hoist tuple dimension, use reorder");
	template<class Original>
	struct dim_replacement {
		static_assert(std::is_same_v<Original, hoisted>, "bug");
		using type = typename Original::ret_sig;
	};
public:
	using signature = function_sig<Dim, typename hoisted::arg_length, typename T::signature::template replace<dim_replacement, Dim>>;

	[[nodiscard]]
	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(state);
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		return offset_of<Sub>(sub_structure(), state);
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		return sub_structure().template length<QDim>(state);
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), state);
	}
};

template<IsDim auto Dim>
struct hoist_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return hoist_t<Dim, Struct>(s); }
};

template<auto Dim, auto ...Dims> requires IsDimPack<decltype(Dim), decltype(Dims)...>
constexpr auto hoist() noexcept {
	if constexpr (sizeof...(Dims) == 0) {
		return hoist_proto<Dim>();
	} else {
		return hoist<Dims...>() ^ hoist_proto<Dim>();
	}
}

namespace helpers {

template<class EvenAcc, class OddAcc, auto ...DimPairs> requires IsDimPack<decltype(DimPairs)...>
struct rename_unzip_dim_pairs;
template<auto ...EvenAcc, auto ...OddAcc, IsDim auto Even, IsDim auto Odd, auto ...DimPairs> requires IsDimPack<decltype(DimPairs)...>
struct rename_unzip_dim_pairs<dim_sequence<EvenAcc...>, dim_sequence<OddAcc...>, Even, Odd, DimPairs...>
	: rename_unzip_dim_pairs<dim_sequence<EvenAcc..., Even>, dim_sequence<OddAcc..., Odd>, DimPairs...> {};
template<class EvenAcc, class OddAcc>
struct rename_unzip_dim_pairs<EvenAcc, OddAcc> {
	using even = EvenAcc;
	using odd = OddAcc;
};

template<class>
struct rename_uniquity;
template<IsDim auto Dim, auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct rename_uniquity<dim_sequence<Dim, Dims...>> {
	static constexpr bool value = (... && (Dims != Dim)) && rename_uniquity<dim_sequence<Dims...>>::value;
};
template<>
struct rename_uniquity<dim_sequence<>> : std::true_type {};

template<IsDim auto QDim, class From, class To>
struct rename_dim;
template<IsDim auto QDim, IsDim auto FromHead, auto ...FromTail, IsDim auto ToHead, auto ...ToTail> requires (QDim != FromHead)
struct rename_dim<QDim, dim_sequence<FromHead, FromTail...>, dim_sequence<ToHead, ToTail...>>
	: rename_dim<QDim, dim_sequence<FromTail...>, dim_sequence<ToTail...>> {};
template<IsDim auto FromHead, auto ...FromTail, IsDim auto ToHead, auto ...ToTail>
struct rename_dim<FromHead, dim_sequence<FromHead, FromTail...>, dim_sequence<ToHead, ToTail...>> {
	static constexpr auto dim = ToHead;
};

template<IsDim auto QDim>
struct rename_dim<QDim, dim_sequence<>, dim_sequence<>> {
	static constexpr auto dim = QDim;
};

template<class From, class To, class Signature>
struct rename_sig;
template<class From, class To, IsDim auto Dim, class ArgLength, class RetSig>
struct rename_sig<From, To, function_sig<Dim, ArgLength, RetSig>> {
	using type = function_sig<rename_dim<Dim, From, To>::dim, ArgLength, typename rename_sig<From, To, RetSig>::type>;
};
template<class From, class To, IsDim auto Dim, class ...RetSigs>
struct rename_sig<From, To, dep_function_sig<Dim, RetSigs...>> {
	using type = dep_function_sig<rename_dim<Dim, From, To>::dim, typename rename_sig<From, To, RetSigs>::type...>;
};
template<class From, class To, class ValueType>
struct rename_sig<From, To, scalar_sig<ValueType>> {
	using type = scalar_sig<ValueType>;
};

template<class From, class To>
struct rename_dim_map {
	template<auto Dim>
	static constexpr auto value = rename_dim<Dim, From, To>::dim;
};

template<class From, class To>
struct rename_state {
	template<class ...StateItem>
	using type = state<state_item<typename StateItem::tag::template map<rename_dim_map<From, To>>, typename StateItem::value_type>...>;

	template<class ...StateItem>
	[[nodiscard]]
	static constexpr auto convert([[maybe_unused]] state<StateItem...> s) noexcept {
		return type<StateItem...>(s.template get<typename StateItem::tag>()...);
	}
};

} // namespace helpers

template<class T, auto ...DimPairs> requires IsDimPack<decltype(DimPairs)...>
struct rename_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static_assert(sizeof...(DimPairs) % 2 == 0, "Expected an even number of dimensions. Usage: rename<Old1, New1, Old2, New2, ...>()");
private:
	using unzip = helpers::rename_unzip_dim_pairs<dim_sequence<>, dim_sequence<>, DimPairs...>;
	using internal = typename unzip::even;
	using external = typename unzip::odd;
	using rename_state = typename helpers::rename_state<external, internal>;
public:
	static_assert(helpers::rename_uniquity<internal>::value, "A dimension is renamed twice. Usage: rename<Old1, New1, Old2, New2, ...>()");
	static_assert(helpers::rename_uniquity<external>::value, "Multiple dimensions are renamed to the same name. Usage: rename<Old1, New1, Old2, New2, ...>()");

	static constexpr char name[] = "rename_t";
	using params = struct_params<
		structure_param<T>,
		dim_param<DimPairs>...>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

private:
	template<class = external, class = internal>
	struct assertion;
	template<auto ...ExternalDims, auto ...InternalDims>
	struct assertion<dim_sequence<ExternalDims...>, dim_sequence<InternalDims...>> {
		template<auto Dim> requires IsDim<decltype(Dim)>
		static constexpr bool is_free = (!T::signature::template any_accept<Dim> || ... || (Dim == InternalDims)); // never used || used but renamed
		static_assert((... && is_free<ExternalDims>), "The structure already has a dimension of a specified name. Usage: rename<Old1, New1, Old2, New2, ...>()");
		// Note: in case a dimension is renamed to itself, is_free returns true. This is necessary to make the above assertion pass.
		// The `rename_uniquity<external>` check already ensures that if a dimension is renamed to itself, no other dimension is renamed to its name.
		static constexpr bool success = true;
	};

	struct filter {
		template<class Tag>
		static constexpr bool value =
			Tag::template any_accept<dim_sequence_contains<external>> ||
			!Tag::template any_accept<dim_sequence_contains<internal>>;
	};

public:
	static_assert(assertion<>::success);
	using signature = typename helpers::rename_sig<internal, external, typename T::signature>::type;

	[[nodiscard]]
	constexpr auto sub_state(IsState auto state) const noexcept {
		return rename_state::convert(state.template filter<filter>());
	}

	[[nodiscard]]
	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		return sub_structure().template length<helpers::rename_dim<QDim, external, internal>::dim>(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<auto ...DimPairs> requires IsDimPack<decltype(DimPairs)...>
struct rename_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return rename_t<Struct, DimPairs...>(s); }
};

template<auto ...DimPairs> requires IsDimPack<decltype(DimPairs)...>
constexpr rename_proto<DimPairs...> rename() noexcept { return {}; }

} // namespace noarr

#endif // NOARR_STRUCTURES_VIEWS_HPP
