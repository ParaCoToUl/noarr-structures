#ifndef NOARR_STRUCTURES_ZCURVE_HPP
#define NOARR_STRUCTURES_ZCURVE_HPP

#include "std_ext.hpp"
#include "structs.hpp"
#include "struct_decls.hpp"
#include "state.hpp"
#include "struct_traits.hpp"
#include "funcs.hpp"

namespace noarr {

namespace helpers {

template<char...>
struct zc_uniquity;
template<char Dim, char... Dims>
struct zc_uniquity<Dim, Dims...> {
	static constexpr bool value = (... && (Dims != Dim)) && zc_uniquity<Dims...>::value;
};
template<>
struct zc_uniquity<> : std::true_type {};

template<class, class>
struct zc_merged_len { using type = dynamic_arg_length; };
template<std::size_t L0, std::size_t L1>
struct zc_merged_len<static_arg_length<L0>, static_arg_length<L1>> { using type = static_arg_length<L0 * L1>; };

template<std::size_t... I, class F>
constexpr auto zc_static_for(std::index_sequence<I...>, F f) noexcept {
	return (... * f(std::integral_constant<std::size_t, I>()));
}

template<std::size_t Levels, class... SizeTs>
constexpr std::tuple<SizeTs...> zc_general(std::size_t z, SizeTs... sizes) noexcept {
	static_assert((... && std::is_same_v<SizeTs, std::size_t>), "bug");
	using EachDim = std::index_sequence_for<SizeTs...>;
	std::tuple<SizeTs...> size = {sizes...};
	std::tuple<SizeTs...> result = {SizeTs(0)...};
	zc_static_for(std::make_index_sequence<Levels>(), [&](auto k) {
		constexpr std::size_t level = Levels - k - 1;
		zc_static_for(EachDim(), [&](auto i) {
			std::size_t small_tile_size = (std::size_t) 1 << level;
			std::size_t facet = zc_static_for(EachDim(), [&](auto j) {
				if constexpr(j == i) {
					return 1;
				} else {
					std::size_t tile_size = small_tile_size;
					if constexpr(j > i)
						tile_size *= 2;
					return (std::get<j>(size) & -tile_size) == std::get<j>(result) ? (std::get<j>(size)-1 & tile_size-1) + 1 : tile_size;
				}
			});
			std::size_t half_volume = facet << level;
			if(z >= half_volume) {
				z -= half_volume;
				std::get<i>(result) += small_tile_size;
			}
			return 0;
		});
		return 0;
	});
	return std::tuple<SizeTs...>(result); // force copy, so that `result` is not aliased due to copy elision
}

template<int Period, std::size_t RepBits = 0, class = void>
struct zc_special_helper {
	using rec = zc_special_helper<2 * Period, (RepBits | RepBits << Period)>;
	static constexpr std::size_t rep_bits = rec::rep_bits;
	static constexpr int num_iter = rec::num_iter + 1;
};
template<int Period, std::size_t RepBits>
struct zc_special_helper<Period, RepBits, std::enable_if_t<Period >= sizeof RepBits * 8>> {
	static constexpr std::size_t rep_bits = RepBits;
	static constexpr int num_iter = 0;
};

template<int NDim, int... I>
constexpr std::size_t zc_special_inner(std::size_t tmp, std::integer_sequence<int, I...>) noexcept {
	(..., (tmp &= zc_special_helper<(NDim << I), ((std::size_t) 1 << (1 << I))-1>::rep_bits, tmp |= tmp >> (NDim-1 << I)));
	return tmp & ((std::size_t) 1 << (1 << sizeof...(I)))-1;
}

template<int NDim, int Dim>
constexpr std::size_t zc_special(std::size_t z) noexcept {
	static_assert(0 <= Dim && Dim < NDim, "bug");
	return zc_special_inner<NDim>(z >> (NDim-Dim-1), std::make_integer_sequence<int, zc_special_helper<NDim>::num_iter>());
}

template<class Acc, char...>
struct zc_dims_pop;
template<char... Acc, char Head, char... Tail>
struct zc_dims_pop<std::integer_sequence<char, Acc...>, Head, Tail...> : zc_dims_pop<std::integer_sequence<char, Acc..., Head>, Tail...> {};
template<char... Acc, char Last>
struct zc_dims_pop<std::integer_sequence<char, Acc...>, Last> {
	static constexpr char dim = Last;
	using dims = std::integer_sequence<char, Acc...>;
};

template<std::size_t N>
struct zc_log2 {
	static_assert(N && !(N & 1), "Z curve size bound and alignment must be powers of two");
	static constexpr int value = zc_log2<(N>>1)>::value + 1;
};
template<>
struct zc_log2<1> {
	static constexpr int value = 0;
};

} // namespace helpers

template<int SpecialLevel, int GeneralLevel, char Dim, class T, char... Dims>
struct merge_zcurve_t : contain<T> {
	using base = contain<T>;
	using base::base;

	// TODO description

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	static_assert(SpecialLevel <= GeneralLevel && GeneralLevel < 8*sizeof(std::size_t), "Invalid parameters");
	static_assert(sizeof...(Dims), "No dimensions to merge");
	static_assert(sizeof...(Dims) <= 8*sizeof(std::size_t), "Too many dimensions to merge");
	static_assert(helpers::zc_uniquity<Dims...>::value, "Cannot merge a dimension with itself");
	static_assert((... && T::signature::template all_accept<Dims>), "The structure does not have a dimension of this name");
	static_assert((... || (Dim == Dims)) || !T::signature::template any_accept<Dim>, "Dimension of this name already exists");
private:
	template<int Remaining, class ArgLenAcc>
	struct dim_replacer {
		template<class Original>
		struct replacement {
			static_assert(!Original::dependent, "Cannot merge a tuple index");
			static_assert(Original::arg_length::is_known, "The dimension lengths must be set before merging");

			static constexpr int remaining = Remaining - 1;
			using arg_len_acc = typename helpers::zc_merged_len<ArgLenAcc, typename Original::arg_length>::type;

			using type = typename Original::ret_sig::replace<dim_replacer<remaining, arg_len_acc>::template replacement, Dims...>;
		};
	};
	template<class ArgLenAcc>
	struct dim_replacer<1, ArgLenAcc> {
		template<class Original>
		struct replacement {
			static_assert(!Original::dependent, "Cannot merge a tuple index");
			static_assert(Original::arg_length::is_known, "The dimension lengths must be set before merging");

			using merged_len = typename helpers::zc_merged_len<ArgLenAcc, typename Original::arg_length>::type;

			using type = function_sig<Dim, merged_len, typename Original::ret_sig>;
		};
	};
	using outer_dim_replacer = dim_replacer<sizeof...(Dims), static_arg_length<1>>;
public:
	using signature = typename T::signature::replace<outer_dim_replacer::template replacement, Dims...>;

	using is = std::make_index_sequence<sizeof...(Dims)>;

	template<class State, std::size_t... DimsI>
	constexpr auto sub_state(State state, std::index_sequence<DimsI...>) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set z-curve length");
		/* TODO if constexpr(Dims != Dim) {
			static_assert(!State::template contains<index_in<Dims>>, "Index in this dimension is overriden by a substructure");
			static_assert(!State::template contains<length_in<Dims>>, "Index in this dimension is overriden by a substructure");
		}...*/
		auto clean_state = state.template remove<index_in<Dim>>();
		if constexpr(State::template contains<index_in<Dim>>) {
			auto index = state.template get<index_in<Dim>>();
			auto index_general = index >> SpecialLevel*sizeof...(Dims);
			auto index_special = index & (1 << SpecialLevel*sizeof...(Dims))-1;
			auto indices = helpers::zc_general<GeneralLevel-SpecialLevel>(index_general, (sub_structure().template length<Dims>(clean_state) >> SpecialLevel)...);
			return clean_state.template with<index_in<Dims>...>(((std::get<DimsI>(indices) << SpecialLevel) + helpers::zc_special<sizeof...(Dims), DimsI>(index_special))...);
		} else {
			return clean_state;
		}
	}

	template<class State>
	constexpr std::size_t size(State state) const noexcept {
		return sub_structure().size(sub_state(state, is()));
	}

	template<class Sub, class State>
	constexpr std::size_t strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "Index has not been set");
		return offset_of<Sub>(sub_structure(), sub_state(state, is()));
	}

	template<char QDim, class State>
	constexpr std::size_t length(State state) const noexcept {
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set z-curve length");
		if constexpr(QDim == Dim) {
			auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>>();
			return (... * sub_structure().template length<Dims>(clean_state));
		} else {
			return sub_structure().template length<QDim>(sub_state(state, is()));
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state, is()));
	}
};

template<int SpecialLevel, int GeneralLevel, char Dim, char... Dims>
struct merge_zcurve_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) noexcept { return merge_zcurve_t<SpecialLevel, GeneralLevel, Dim, Struct, Dims...>(s); }
};

template<char... AllDims>
struct merge_zcurve {
private:
	using dims_pop = helpers::zc_dims_pop<std::integer_sequence<char>, AllDims...>;
	struct error { static_assert(always_false<merge_zcurve<AllDims...>>, "Do not instantiate this type directly, use merge_zcurve<'original dims', 'new dim'>::maxsize_alignment<size, alignment>()"); };

public:
	template<class = error>
	merge_zcurve(error = {});

	template<std::size_t Size, std::size_t Alignment>
	static constexpr auto maxsize_alignment() noexcept {
		return maxsize_alignment<helpers::zc_log2<Alignment>::value, helpers::zc_log2<Size>::value, dims_pop::dim>(typename dims_pop::dims());
	}

private:
	template<int SpecialLevel, int GeneralLevel, char Dim, char... Dims>
	static constexpr merge_zcurve_proto<SpecialLevel, GeneralLevel, Dim, Dims...> maxsize_alignment(std::integer_sequence<char, Dims...>) noexcept {
		return {};
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_ZCURVE_HPP
