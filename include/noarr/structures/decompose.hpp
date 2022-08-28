#ifndef NOARR_STRUCTURES_DECOMPOSE_HPP
#define NOARR_STRUCTURES_DECOMPOSE_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "struct_getters.hpp"
#include "funcs.hpp"

namespace noarr {

template<char Dim, char DimMajor, char DimMinor, class T>
struct decompose_t : contain<T> {
	using base = contain<T>;
	using base::base;

	constexpr std::tuple<T> sub_structures() const noexcept { return std::tuple<T>(base::template get<0>()); }
	using description = struct_description<
		char_pack<'d', 'e', 'c', 'o', 'm', 'p', 'o', 's', 'e'>,
		dims_impl<DimMajor, DimMinor>,
		dims_impl<Dim>,
		structure_param<T>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	static_assert(DimMajor != DimMinor, "Cannot use the same name for both components of a dimension");
	static_assert(T::struct_type::template all_accept<Dim>, "The structure does not have a dimension of this name");
	static_assert(DimMajor == Dim || !T::struct_type::template any_accept<DimMajor>, "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::struct_type::template any_accept<DimMinor>, "Dimension of this name already exists");
private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot decompose a tuple index");
		using major_length = std::conditional_t<Original::arg_length::is_known, dynamic_arg_length, unknown_arg_length>;
		using minor_length = unknown_arg_length;
		using type = function_type<DimMajor, major_length, function_type<DimMinor, minor_length, typename Original::ret_type>>;
	};
public:
	using struct_type = typename T::struct_type::replace<dim_replacement, Dim>;

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		// TODO check and translate
		return sub_structure().size(state);
	}
};

template<char Dim, char DimMajor, char DimMinor>
struct decompose_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return decompose_t<Dim, DimMajor, DimMinor, Struct>(s); }
};

template<char Dim, char DimMajor, char DimMinor>
constexpr auto decompose() {
	return decompose_proto<Dim, DimMajor, DimMinor>();
}

template<char Dim, char DimMajor, char DimMinor, class MinorSizeT>
constexpr auto decompose(MinorSizeT minor_length) {
	return decompose_proto<Dim, DimMajor, DimMinor>() ^ set_length<DimMinor>(minor_length);
}



template<char Dim, char DimMajor, char DimMinor, class T>
struct spi_offset<decompose_t<Dim, DimMajor, DimMinor, T>> {
	template<class State>
	static constexpr std::size_t get(const decompose_t<Dim, DimMajor, DimMinor, T> &view, State state) {
		auto major_index = state.template get<index_in<DimMajor>>();
		auto minor_index = state.template get<index_in<DimMinor>>();
		auto minor_length = state.template get<length_in<DimMinor>>();
		auto tmp_state = state
			.template remove<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>()
			.template with<index_in<Dim>>(major_index*minor_length + minor_index);
		if constexpr(State::template contains<length_in<DimMajor>>) {
			auto major_length = state.template get<length_in<DimMajor>>();
			auto sub_state = tmp_state.template with<length_in<Dim>>(major_length*minor_length);
			return spi_offset_get(view.sub_structure(), sub_state);
		} else {
			return spi_offset_get(view.sub_structure(), tmp_state);
		}
	}
};

template<char QDim, char Dim, char DimMajor, char DimMinor, class T>
struct spi_length<QDim, decompose_t<Dim, DimMajor, DimMinor, T>> {
	template<class State>
	static constexpr std::size_t get(const decompose_t<Dim, DimMajor, DimMinor, T> &view, State state) {
		if constexpr(QDim == DimMinor) {
			static_assert(helpers::wrong_dim<QDim>, "Length has not been set");
			return 0;
		} else if constexpr(QDim == DimMajor) {
			auto minor_length = state.template get<length_in<DimMinor>>();
			auto sub_state = state.template remove<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>();
			return spi_length_get<Dim>(view.sub_structure(), sub_state) / minor_length;
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			if constexpr(State::template contains<index_in<DimMajor>> && State::template contains<index_in<DimMinor>>) {
				auto major_index = state.template get<index_in<DimMajor>>();
				auto minor_index = state.template get<index_in<DimMinor>>();
				auto minor_length = state.template get<length_in<DimMinor>>();
				auto sub_state = state
					.template remove<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>()
					.template with<index_in<Dim>>(major_index*minor_length + minor_index);
				return spi_length_get<QDim>(view.sub_structure(), sub_state);
			} else {
				auto sub_state = state.template remove<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>();
				return spi_length_get<QDim>(view.sub_structure(), sub_state);
			}
		}
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_DECOMPOSE_HPP
