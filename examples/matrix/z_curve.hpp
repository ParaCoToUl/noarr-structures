#ifndef Z_CURVE_HPP
#define Z_CURVE_HPP

#include "noarr/structures/structs.hpp"
#include "noarr/structures/struct_traits.hpp"

namespace noarr {

namespace helpers {

template<char Dim>
struct z_curve_bottom : private contain<std::size_t> {
    using base = contain<std::size_t>;
	using description = struct_description<
		char_pack<'z','_','c','u','r','v','e','_','b','o','t','t','o','m'>,
		dims_impl<Dim>,
		dims_impl<>>;

	constexpr z_curve_bottom() = default;
	constexpr z_curve_bottom(std::size_t length) : base(length) {};

	constexpr auto construct() const {
		return z_curve_bottom<Dim>();
	}

	static constexpr std::size_t size() { return 0; }
    constexpr std::size_t length() const { return base::template get<0>(); }
	static constexpr std::size_t offset(std::uint16_t i) {
		i = (i | (i << 8)) & 0x00FF00FF;
		i = (i | (i << 4)) & 0x0F0F0F0F;
		i = (i | (i << 2)) & 0x33333333;
		i = (i | (i << 1)) & 0x55555555;

		return i;
	}
};

template<typename T, typename... KS>
struct z_curve_top_get_t;

template<typename T>
struct z_curve_top_get_t<T> {
	using type = T;
};

template<typename T>
struct z_curve_top_get_t<T, void> {
	using type = T;
};

template<typename T, typename TH1, typename TH2>
struct z_curve_top : private contain<T, TH1, TH2> {
	using base = contain<T, TH1, TH2>;
	constexpr auto sub_structures() const {
		return std::tuple_cat(base::template get<0>().sub_structures(), std::make_tuple(base::template get<1>(), base::template get<2>()));
	}

	using description = struct_description<
		char_pack<'z','_','c','u','r','v','e','_','t','o','p'>,
		dims_impl<>,
		dims_impl<>,
		type_param<T>,
		type_param<TH1>,
		type_param<TH2>>;
	
	template<typename... KS>
	using get_t = typename z_curve_top_get_t<T, KS...>::type;

	constexpr z_curve_top() = default;
	explicit constexpr z_curve_top(T sub_structure, TH1 sub_structure1, TH2 sub_structure2) : base(sub_structure, sub_structure1, sub_structure2) {}

	template<typename T2, typename TH3, typename TH4>
	constexpr auto construct(T2 sub_structure, TH3 sub_structure1, TH4 sub_structure2) const {
		return z_curve_top<decltype(this->base::template get<0>().construct(sub_structure)), TH3, TH4>(base::template get<0>().construct(sub_structure), sub_structure1, sub_structure2);
	}

	constexpr std::size_t size() const { return base::template get<0>().size(); }
	constexpr std::size_t offset() const {
		return base::template get<0>().offset(base::template get<1>().offset() | (base::template get<2>().offset() << 1));
	}
};

}

template<char Dim1, char Dim2, typename T>
using z_curve = helpers::z_curve_top<T, helpers::z_curve_bottom<Dim1>, helpers::z_curve_bottom<Dim2>>;

}

#endif // Z_CURVE_HPP
