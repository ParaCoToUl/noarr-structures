#ifndef NOARR_STRUCTURES_DECOMPOSE_HPP
#define NOARR_STRUCTURES_DECOMPOSE_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "struct_getters.hpp"
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
	static_assert(T::struct_type::template all_accept<DimMajor>, "The structure does not have a dimension of this name");
	static_assert(T::struct_type::template all_accept<DimMinor>, "The structure does not have a dimension of this name");
	static_assert(Dim == DimMajor || Dim == DimMinor || !T::struct_type::template any_accept<Dim>, "Dimension of this name already exists");
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

			using type = function_type<Dim, typename composed_len<>::type, typename OriginalInner::ret_type>;
		};

		using type = typename Original::ret_type::replace<inner_dim_replacement, DimMinor, DimMajor>;
	};
public:
	using struct_type = typename T::struct_type::replace<outer_dim_replacement, DimMajor, DimMinor>;
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



template<char DimMajor, char DimMinor, char Dim, class T>
struct spi_size<compose_t<DimMajor, DimMinor, Dim, T>> {
	template<class State>
	static constexpr std::size_t get(const compose_t<DimMajor, DimMinor, Dim, T> &view, State state) {
		return spi_size_get(view.sub_structure(), state);
	}
};

template<char DimMajor, char DimMinor, char Dim, class T>
struct spi_offset<compose_t<DimMajor, DimMinor, Dim, T>> {
	template<class State>
	static constexpr std::size_t get(const compose_t<DimMajor, DimMinor, Dim, T> &view, State state) {
		auto tmp_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		auto minor_length = spi_length_get<DimMinor>(view.sub_structure(), tmp_state);
		auto index = state.template get<index_in<Dim>>();
		auto sub_state = tmp_state.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length, index % minor_length);
		if constexpr(State::template contains<length_in<Dim>>) {
			auto length = state.template get<length_in<Dim>>();
			return spi_offset_get(view.sub_structure(), sub_state.template with<length_in<DimMajor>>(length / minor_length));
		} else {
			return spi_offset_get(view.sub_structure(), sub_state);
		}
	}
};

template<char QDim, char DimMajor, char DimMinor, char Dim, class T>
struct spi_length<QDim, compose_t<DimMajor, DimMinor, Dim, T>> {
	template<class State>
	static constexpr std::size_t get(const compose_t<DimMajor, DimMinor, Dim, T> &view, State state) {
		auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		if constexpr(QDim == Dim) {
			return spi_length_get<DimMajor>(view.sub_structure(), sub_state) * spi_length_get<DimMinor>(view.sub_structure(), sub_state);
		} else {
			return spi_length_get<QDim>(view.sub_structure(), sub_state);
		}
	}
};



template<char DimMajor, char DimMinor, char Dim, class T>
struct spi_traits<compose_t<DimMajor, DimMinor, Dim, T>> {
	template<class State>
	static auto get(const compose_t<DimMajor, DimMinor, Dim, T> &view, State state) {
		auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		return spi_traits_get(view.sub_structure(), sub_state);
	}
};

template<char DimMajor, char DimMinor, char Dim, class T>
struct spi_type<compose_t<DimMajor, DimMinor, Dim, T>> {
	template<class State>
	static auto get(const compose_t<DimMajor, DimMinor, Dim, T> &view, State state) {
		auto sub_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		return spi_type_get(view.sub_structure(), sub_state);
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_DECOMPOSE_HPP
