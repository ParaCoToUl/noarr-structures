#ifndef NOARR_STRUCTURES_DECOMPOSE_HPP
#define NOARR_STRUCTURES_DECOMPOSE_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "struct_decls.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "funcs.hpp"

namespace noarr {

template<char DimMajor, char DimMinor, char Dim, class T>
struct compose_t : contain<T> {
	using base = contain<T>;
	using base::base;

	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(base::template get<0>()); }
	using description = struct_description<
		char_pack<'c', 'o', 'm', 'p', 'o', 's', 'e'>,
		dims_impl<Dim>,
		dims_impl<DimMajor, DimMinor>,
		structure_param<T>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	static_assert(DimMajor != DimMinor, "Cannot compose a dimension with itself");
	static_assert(T::signature::template all_accept<DimMajor>, "The structure does not have a dimension of this name");
	static_assert(T::signature::template all_accept<DimMinor>, "The structure does not have a dimension of this name");
	static_assert(Dim == DimMajor || Dim == DimMinor || !T::signature::template any_accept<Dim>, "Dimension of this name already exists");
private:
	template<class Original>
	struct outer_dim_replacement {
		static_assert(!Original::dependent, "Cannot compose a tuple index");
		template<class OriginalInner>
		struct inner_dim_replacement {
			static_assert(!OriginalInner::dependent, "Cannot compose a tuple index");
			using OrigMajor = std::conditional_t<Original::dim == DimMajor, Original, OriginalInner>;
			using OrigMinor = std::conditional_t<Original::dim == DimMinor, Original, OriginalInner>;
			static_assert(OrigMajor::dim == DimMajor && OrigMinor::dim == DimMinor, "bug");
			static_assert(OrigMinor::arg_length::is_known, "The minor dimension length must be set before composition");

			template<bool = OrigMinor::arg_length::is_static && OrigMajor::arg_length::is_static, bool = OrigMajor::arg_length::is_known, class = void>
			struct composed_len;
			template<class Useless>
			struct composed_len<true, true, Useless> { using type = static_arg_length<OrigMajor::arg_length::value * OrigMinor::arg_length::value>; };
			template<class Useless>
			struct composed_len<false, true, Useless> { using type = dynamic_arg_length; };
			template<class Useless>
			struct composed_len<false, false, Useless> { using type = unknown_arg_length; };

			using type = function_sig<Dim, typename composed_len<>::type, typename OriginalInner::ret_sig>;
		};

		using type = typename Original::ret_sig::replace<inner_dim_replacement, DimMinor, DimMajor>;
	};
public:
	using signature = typename T::signature::replace<outer_dim_replacement, DimMajor, DimMinor>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		// TODO check and translate
		return sub_structure().size(state);
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		// TODO check
		auto tmp_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		auto minor_length = sub_structure().template length<DimMinor>(tmp_state);
		auto index = state.template get<index_in<Dim>>();
		auto sub_state = tmp_state.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length, index % minor_length);
		if constexpr(State::template contains<length_in<Dim>>) {
			auto length = state.template get<length_in<Dim>>();
			return offset_of<Sub>(sub_structure(), sub_state.template with<length_in<DimMajor>>(length / minor_length));
		} else {
			return offset_of<Sub>(sub_structure(), sub_state);
		}
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		// TODO use length?
		auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		if constexpr(QDim == Dim) {
			return sub_structure().template length<DimMajor>(sub_state) * sub_structure().template length<DimMinor>(sub_state);
		} else {
			return sub_structure().template length<QDim>(sub_state);
		}
	}
};

template<char DimMajor, char DimMinor, char Dim>
struct compose_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return compose_t<DimMajor, DimMinor, Dim, Struct>(s); }
};

template<char DimMajor, char DimMinor, char Dim>
constexpr auto compose() {
	return compose_proto<DimMajor, DimMinor, Dim>();
}

template<char DimMajor, char DimMinor, char Dim, class MinorSizeT>
constexpr auto compose(MinorSizeT minor_length) {
	return set_length<DimMinor>(minor_length) ^ compose_proto<DimMajor, DimMinor, Dim>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_DECOMPOSE_HPP
