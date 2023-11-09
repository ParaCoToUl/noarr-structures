#ifndef NOARR_STRUCTURES_SETTERS_HPP
#define NOARR_STRUCTURES_SETTERS_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<IsDim auto Dim, class T, class IdxT>
struct fix_t : strict_contain<T, IdxT> {
	using strict_contain<T, IdxT>::strict_contain;

	static constexpr char name[] = "fix_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<IdxT>>;

	constexpr T sub_structure() const noexcept { return this->template get<0>(); }
	constexpr IdxT idx() const noexcept { return this->template get<1>(); }

private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		static_assert(ArgLength::is_known, "Index cannot be fixed until its length is set");
		using type = RetSig;
	};
	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(IdxT::value || true, "Tuple index must be set statically, wrap it in lit<> (e.g. replace 42 with lit<42>)");
		using type = typename original::template ret_sig<IdxT::value>;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	constexpr auto sub_state(IsState auto state) const noexcept {
		return state.template remove<length_in<Dim>>().template with<index_in<Dim>>(idx());
	}

	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub>
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<IsDim auto QDim>
	constexpr auto length(IsState auto state) const noexcept {
		static_assert(QDim != Dim, "This dimension is already fixed, it cannot be used from outside");
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub>
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class IdxT>
struct fix_proto : strict_contain<IdxT> {
	using strict_contain<IdxT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return fix_t<Dim, Struct, IdxT>(s, this->get()); }
};

/**
 * @brief fixes an index (or indices) given by dimension name(s) in a structure
 *
 * @tparam Dims: the dimension names
 * @param ts: parameters for fixing the indices
 */
template<auto... Dim, class... IdxT> requires ((sizeof...(Dim) == sizeof...(IdxT)) && ... && IsDim<decltype(Dim)>)
constexpr auto fix(IdxT... idx) noexcept {
	if constexpr (sizeof...(Dim) > 0)
		return (... ^ fix_proto<Dim, good_index_t<IdxT>>(idx));
	else
		return neutral_proto();
}

template<IsDim auto Dim, class T, class LenT>
struct set_length_t : strict_contain<T, LenT> {
	using strict_contain<T, LenT>::strict_contain;

	static constexpr char name[] = "set_length_t";
	using params = struct_params<
		dim_param<Dim>,
		structure_param<T>,
		type_param<LenT>>;

	constexpr T sub_structure() const noexcept { return this->template get<0>(); }
	constexpr LenT len() const noexcept { return this->template get<1>(); }

private:
	template<class Original>
	struct dim_replacement;
	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		static_assert(!ArgLength::is_known, "The length in this dimension is already set");
		using type = function_sig<Dim, arg_length_from_t<LenT>, RetSig>;
	};
	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		static_assert(value_always_false<Dim>, "Cannot set tuple length");
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	constexpr auto sub_state(IsState auto state) const noexcept {
		return state.template with<length_in<Dim>>(len());
	}

	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub>
	constexpr auto strict_offset_of(IsState auto state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<IsDim auto QDim>
	constexpr auto length(IsState auto state) const noexcept {
		return sub_structure().template length<QDim>(sub_state(state));
	}

	template<class Sub>
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class LenT>
struct set_length_proto : strict_contain<LenT> {
	using strict_contain<LenT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return set_length_t<Dim, Struct, LenT>(s, this->get()); }
};

/**
 * @brief sets the length of a `vector` specified by the dimension name
 *
 * @tparam Dim: the dimension name of the transformed structure
 * @param length: the desired length
 */
template<auto... Dim, class... LenT> requires (... && IsDim<decltype(Dim)>)
constexpr auto set_length(LenT... len) noexcept { return (... ^ set_length_proto<Dim, good_index_t<LenT>>(len)); }

template<>
constexpr auto set_length<>() noexcept { return neutral_proto(); }

} // namespace noarr

#endif // NOARR_STRUCTURES_SETTERS_HPP
