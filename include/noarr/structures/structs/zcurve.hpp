#ifndef NOARR_STRUCTURES_ZCURVE_HPP
#define NOARR_STRUCTURES_ZCURVE_HPP

#include <cstddef>
#include <bit>
#include <tuple>
#include <type_traits>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

namespace helpers {

template<auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct zc_uniquity;
template<IsDim auto Dim, auto ...Dims> requires IsDimPack<decltype(Dims)...>
struct zc_uniquity<Dim, Dims...> {
	static constexpr bool value = (... && (Dims != Dim)) && zc_uniquity<Dims...>::value;
};
template<>
struct zc_uniquity<> : std::true_type {};

template<class, class>
struct zc_merged_len { using type = dynamic_arg_length; };
template<std::size_t L0, std::size_t L1>
struct zc_merged_len<static_arg_length<L0>, static_arg_length<L1>> { using type = static_arg_length<L0 * L1>; };

// std::integral_constant but less annoying
template<std::size_t Value>
struct zc_constexpr { static constexpr auto v = Value; };

template<std::size_t ...I, class F>
constexpr void zc_static_for([[maybe_unused]] std::index_sequence<I...> is, F f) noexcept {
	(..., f(zc_constexpr<I>()));
}

template<std::size_t ...I, class F>
constexpr auto zc_product_static_for([[maybe_unused]] std::index_sequence<I...> is, F f) noexcept {
	return (... * f(zc_constexpr<I>()));
}

template<std::size_t Levels, class ...SizeTs>
constexpr std::tuple<SizeTs...> zc_general(std::size_t z, SizeTs ...sizes) noexcept {
	static_assert((... && std::is_same_v<SizeTs, std::size_t>), "bug");
	using EachDim = std::index_sequence_for<SizeTs...>;
	std::tuple<SizeTs...> size = {sizes...};
	std::tuple<SizeTs...> result = {SizeTs(0)...};
	zc_static_for(std::make_index_sequence<Levels>(), [&]<class k>(k) {
		zc_static_for(EachDim(), [&]<class i>(i) {
			using level = zc_constexpr<Levels - k::v - 1U>;
			using small_tile_size = zc_constexpr<static_cast<std::size_t>(1U) << level::v>;
			const std::size_t facet = zc_product_static_for(EachDim(), [&]<class j>(j) {
				if constexpr(j::v == i::v) {
					return 1U;
				} else {
					constexpr std::size_t tile_size = (j::v > i::v ? 2 : 1U) * small_tile_size::v;
					return (std::get<j::v>(size) & -tile_size) == std::get<j::v>(result) ? ((std::get<j::v>(size) - 1U) & (tile_size - 1U)) + 1U : tile_size;
				}
			});
			const std::size_t half_volume = facet << level::v;
			if(z >= half_volume) {
				z -= half_volume;
				std::get<i::v>(result) += small_tile_size::v;
			}
		});
	});
	return std::tuple<SizeTs...>(result); // force copy, so that `result` is not aliased due to copy elision
}

template<std::size_t Period, std::size_t RepBits = 0>
struct zc_special_helper {
	using rec = zc_special_helper<2 * Period, (RepBits | RepBits << Period)>;
	static constexpr std::size_t rep_bits = rec::rep_bits;
	static constexpr std::size_t num_iter = rec::num_iter + 1U;
};
template<std::size_t Period, std::size_t RepBits> requires (Period >= sizeof RepBits * 8)
struct zc_special_helper<Period, RepBits> {
	static constexpr std::size_t rep_bits = RepBits;
	static constexpr std::size_t num_iter = 0;
};

template<std::size_t NDim, std::size_t ...I>
constexpr std::size_t zc_special_inner(std::size_t tmp, [[maybe_unused]] std::integer_sequence<std::size_t, I...> is) noexcept {
	(..., (tmp &= zc_special_helper<(NDim << I), (static_cast<std::size_t>(1U) << (1U << I)) - 1U>::rep_bits, tmp |= tmp >> ((NDim - 1U) << I)));
	return tmp & ((static_cast<std::size_t>(1U) << (1U << sizeof...(I))) - 1U);
}

template<std::size_t NDim, std::size_t Dim>
constexpr std::size_t zc_special(std::size_t z) noexcept {
	static_assert(0 <= Dim && Dim < NDim, "bug");
	return zc_special_inner<NDim>(z >> (NDim-Dim-1U), std::make_integer_sequence<std::size_t, zc_special_helper<NDim>::num_iter>());
}

template<class Acc, auto...>
struct zc_dims_pop;
template<auto ...Acc, IsDim auto Head, auto ...Tail>
struct zc_dims_pop<dim_sequence<Acc...>, Head, Tail...> : zc_dims_pop<dim_sequence<Acc..., Head>, Tail...> {};
template<auto ...Acc, IsDim auto Last>
struct zc_dims_pop<dim_sequence<Acc...>, Last> {
	static constexpr auto dim = Last;
	using dims = dim_sequence<Acc...>;
};

} // namespace helpers

template<std::size_t SpecialLevel, std::size_t GeneralLevel, IsDim auto Dim, class T, auto ...Dims>
struct merge_zcurve_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "merge_zcurve_t";
	using params = struct_params<
		value_param<SpecialLevel>,
		value_param<GeneralLevel>,
		dim_param<Dim>,
		structure_param<T>,
		dim_param<Dims>...>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

	static_assert(SpecialLevel <= GeneralLevel && GeneralLevel < 8*sizeof(std::size_t), "Invalid parameters");
	static_assert(sizeof...(Dims), "No dimensions to merge");
	static_assert(sizeof...(Dims) <= 8*sizeof(std::size_t), "Too many dimensions to merge");
	static_assert(helpers::zc_uniquity<Dims...>::value, "Cannot merge a dimension with itself");
	static_assert((... || (Dim == Dims)) || !T::signature::template any_accept<Dim>, "Dimension of this name already exists");
private:
	template<std::size_t Remaining, class ArgLenAcc>
	struct dim_replacer {
		template<class Original>
		struct replacement {
			static_assert(!Original::dependent, "Cannot merge a tuple index");
			static_assert(Original::arg_length::is_known, "The dimension lengths must be set before merging");

			static constexpr std::size_t remaining = Remaining - 1U;
			using arg_len_acc = typename helpers::zc_merged_len<ArgLenAcc, typename Original::arg_length>::type;

			using type = typename Original::ret_sig::template replace<dim_replacer<remaining, arg_len_acc>::template replacement, Dims...>;
		};
	};
	template<class ArgLenAcc>
	struct dim_replacer<1U, ArgLenAcc> {
		template<class Original>
		struct replacement {
			static_assert(!Original::dependent, "Cannot merge a tuple index");
			static_assert(Original::arg_length::is_known, "The dimension lengths must be set before merging");

			using merged_len = typename helpers::zc_merged_len<ArgLenAcc, typename Original::arg_length>::type;

			using type = function_sig<Dim, merged_len, typename Original::ret_sig>;
		};
	};
	using outer_dim_replacer = dim_replacer<sizeof...(Dims), static_arg_length<1U>>;
public:
	using signature = typename T::signature::template replace<outer_dim_replacer::template replacement, Dims...>;

	using is = std::make_index_sequence<sizeof...(Dims)>;

	template<std::size_t ...DimsI, class State> requires (sizeof...(DimsI) == sizeof...(Dims) && IsState<State>)
	[[nodiscard]]
	constexpr auto sub_state(State state, [[maybe_unused]] std::index_sequence<DimsI...> is) const noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set z-curve length");
		const auto clean_state = state.template remove<index_in<Dim>, index_in<Dims>..., length_in<Dims>...>();
		if constexpr(State::template contains<index_in<Dim>>) {
			const std::size_t index = state.template get<index_in<Dim>>();
			const auto index_general = index >> SpecialLevel*sizeof...(Dims);
			const auto index_special = index & ((1U << SpecialLevel*sizeof...(Dims)) - 1U);
			const auto indices = helpers::zc_general<GeneralLevel-SpecialLevel>(index_general, (sub_structure().template length<Dims>(clean_state) >> SpecialLevel)...);
			return clean_state.template with<index_in<Dims>...>(((std::get<DimsI>(indices) << SpecialLevel) + helpers::zc_special<sizeof...(Dims), DimsI>(index_special))...);
		} else {
			return clean_state;
		}
	}

	[[nodiscard]]
	constexpr auto size(IsState auto state) const noexcept {
		return sub_structure().size(sub_state(state, is()));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "Index has not been set");
		return offset_of<Sub>(sub_structure(), sub_state(state, is()));
	}

	template<auto QDim, IsState State> requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set z-curve length");
		if constexpr(QDim == Dim) {
			return (... * sub_structure().template length<Dims>(sub_state(state, is())));
		} else {
			return sub_structure().template length<QDim>(sub_state(state, is()));
		}
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state, is()));
	}
};

template<std::size_t SpecialLevel, std::size_t GeneralLevel, IsDim auto Dim, auto ...Dims>
struct merge_zcurve_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return merge_zcurve_t<SpecialLevel, GeneralLevel, Dim, Struct, Dims...>(s); }
};

template<auto ...AllDims> requires IsDimPack<decltype(AllDims)...>
struct merge_zcurve {
private:
	using dims_pop = helpers::zc_dims_pop<dim_sequence<>, AllDims...>;
	struct error { static_assert(always_false<merge_zcurve<AllDims...>>, "Do not instantiate this type directly, use merge_zcurve<'original dims', 'new dim'>::maxlen_alignment<len, alignment>()"); };

public:
	template<class = error>
	explicit merge_zcurve(error = {});

	template<std::size_t MaxLen, std::size_t Alignment> requires (std::popcount(MaxLen) == 1U && std::popcount(Alignment) == 1U) // must be powers of 2
	[[nodiscard]]
	static constexpr auto maxlen_alignment() noexcept {
		return maxlen_alignment<std::countr_zero(Alignment), std::countr_zero(MaxLen), dims_pop::dim>(typename dims_pop::dims());
	}

private:
	template<std::size_t SpecialLevel, std::size_t GeneralLevel, auto Dim, auto ...Dims> requires (IsDim<decltype(Dim)> && ... && IsDim<decltype(Dims)>)
	static constexpr merge_zcurve_proto<SpecialLevel, GeneralLevel, Dim, Dims...> maxlen_alignment([[maybe_unused]] dim_sequence<Dims...> ds) noexcept {
		return {};
	}
};

} // namespace noarr

#endif // NOARR_STRUCTURES_ZCURVE_HPP
