#ifndef NOARR_STRUCTURES_SLICE_HPP
#define NOARR_STRUCTURES_SLICE_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

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
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return shift_t<Dim, Struct, StartT>(s, base::template get<0>()); }
};

/**
 * @brief shifts an index (or indices) given by dimension name(s) in a structure
 * 
 * @tparam Dim: the dimension names
 * @param start: parameters for shifting the indices
 */
template<char... Dim, class... StartT>
constexpr auto shift(StartT... start) noexcept { return (... ^ shift_proto<Dim, good_index_t<StartT>>(start)); }

template<>
constexpr auto shift<>() noexcept { return neutral_proto(); }

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
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return slice_t<Dim, Struct, StartT, LenT>(s, base::template get<0>(), base::template get<1>()); }
};

template<char Dim, class StartT, class LenT>
constexpr auto slice(StartT start, LenT len) noexcept { return slice_proto<Dim, good_index_t<StartT>, good_index_t<LenT>>(start, len); }

} // namespace noarr

#endif // NOARR_STRUCTURES_SLICE_HPP
