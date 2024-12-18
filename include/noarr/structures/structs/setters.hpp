#ifndef NOARR_STRUCTURES_SETTERS_HPP
#define NOARR_STRUCTURES_SETTERS_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<auto Dim, class T, class IdxT> requires IsDim<decltype(Dim)>
struct fix_t : strict_contain<T, IdxT> {
	using strict_contain<T, IdxT>::strict_contain;

	static constexpr char name[] = "fix_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<IdxT>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->template get<0>(); }

	[[nodiscard]]
	constexpr IdxT idx() const noexcept { return this->template get<1>(); }

private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		using type = RetSig;
	};
	template<class ...RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(((void)IdxT::value, true), "Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
		using type = typename original::template ret_sig<IdxT::value>;
	};

	template<IsState State>
	[[nodiscard]]
	static constexpr auto sub_state_impl(State state, IdxT idx) noexcept {
		return state.template remove<length_in<Dim>>().template with<index_in<Dim>>(idx);
	}

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return sub_state_impl(state, idx());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(sub_state_impl(std::declval<State>(), std::declval<IdxT>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>()) {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>()) {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, fix_t, State>()) {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr(QDim == Dim) {
			return false;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept
	requires (has_length<QDim, State>()) {
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, fix_t, State>()) {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<auto Dim, class IdxT> requires IsDim<decltype(Dim)>
struct fix_proto : strict_contain<IdxT> {
	using strict_contain<IdxT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return fix_t<Dim, Struct, IdxT>(s, this->get()); }
};

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 *
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<auto ...Dims, class ...IdxT> requires (sizeof...(Dims) == sizeof...(IdxT)) && IsDimPack<decltype(Dims)...>
constexpr auto fix(IdxT ...idx) noexcept {
	if constexpr (sizeof...(Dims) > 0) {
		return (... ^ fix_proto<Dims, good_index_t<IdxT>>(idx));
	} else {
		return neutral_proto();
	}
}

template<auto Dim, class T, class LenT> requires IsDim<decltype(Dim)>
struct set_length_t : strict_contain<T, LenT> {
	using strict_contain<T, LenT>::strict_contain;

	static constexpr char name[] = "set_length_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<LenT>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->template get<0>(); }
	[[nodiscard]]
	constexpr LenT len() const noexcept { return this->template get<1>(); }

private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		using type = function_sig<Dim, arg_length_from_t<LenT>, RetSig>;
	};
	template<class ...RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		static_assert(value_always_false<Dim>, "Cannot set tuple length");
	};

	template<IsState State>
	[[nodiscard]]
	static constexpr auto sub_state_impl(State state, LenT len) noexcept {
		return state.template with<length_in<Dim>>(len);
	}

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return sub_state_impl(state, len());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(sub_state_impl(std::declval<State>(), std::declval<LenT>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>()) {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>()) {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, set_length_t, State>()) {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set length of a dimension twice");
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept
	requires (has_length<QDim, State>()) {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set length of a dimension twice");
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, set_length_t, State>()) {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<auto Dim, class LenT> requires IsDim<decltype(Dim)>
struct set_length_proto : strict_contain<LenT> {
	using strict_contain<LenT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return set_length_t<Dim, Struct, LenT>(s, this->get()); }
};

/**
 * @brief sets the length of a `vector` specified by the dimension name
 *
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<auto ...Dims, class ...LenT> requires IsDimPack<decltype(Dims)...>
constexpr auto set_length(LenT ...len) noexcept { return (... ^ set_length_proto<Dims, good_index_t<LenT>>(len)); }

template<>
constexpr auto set_length<>() noexcept { return neutral_proto(); }

} // namespace noarr

#endif // NOARR_STRUCTURES_SETTERS_HPP
