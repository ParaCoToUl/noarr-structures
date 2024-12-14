#ifndef NOARR_STRUCTURES_BLOCKS_HPP
#define NOARR_STRUCTURES_BLOCKS_HPP

#include <cstddef>
#include <type_traits>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T>
struct into_blocks_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "into_blocks_t";
	using params = struct_params<
		dim_param<Dim>,
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		structure_param<T>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

	static_assert(DimMajor != DimMinor, "Cannot use the same name for both components of a dimension");
	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>, "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>, "Dimension of this name already exists");
private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");
		using major_length = std::conditional_t<Original::arg_length::is_known, dynamic_arg_length, unknown_arg_length>;
		using minor_length = unknown_arg_length;
		using type = function_sig<DimMajor, major_length, function_sig<DimMinor, minor_length, typename Original::ret_sig>>;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		const auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>();
		constexpr bool have_indices = State::template contains<index_in<DimMajor>, index_in<DimMinor>>;
		if constexpr(State::template contains<length_in<DimMajor>, length_in<DimMinor>>) {
			const auto major_length = state.template get<length_in<DimMajor>>();
			const auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<length_in<Dim>, index_in<Dim>>(major_length*minor_length, major_index*minor_length + minor_index);
			} else {
				return clean_state.template with<length_in<Dim>>(major_length*minor_length);
			}
		} else if constexpr(State::template contains<length_in<DimMinor>>) {
			const auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<index_in<Dim>>(major_index*minor_length + minor_index);
			} else {
				return clean_state;
			}
		} else {
			return clean_state;
		}
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
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == DimMinor) {
			static_assert(State::template contains<length_in<DimMinor>>, "Length has not been set");
			return state.template get<length_in<DimMinor>>();
		} else if constexpr(QDim == DimMajor) {
			if constexpr(State::template contains<length_in<DimMajor>>) {
				return state.template get<length_in<DimMajor>>();
			} else if constexpr(State::template contains<length_in<DimMinor>>) {
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto full_length = sub_structure().template length<Dim>(sub_state(state));
				return full_length / minor_length;
			} else {
				static_assert(value_always_false<QDim>, "Length has not been set (and cannot be computed from the total size because block size has also not been set)");
			}
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor>
struct into_blocks_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return into_blocks_t<Dim, DimMajor, DimMinor, Struct>(s); }
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor = Dim>
constexpr auto into_blocks() {
	return into_blocks_proto<Dim, DimMajor, DimMinor>();
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T>
struct into_blocks_dynamic_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "into_blocks_dynamic_t";
	using params = struct_params<
		dim_param<Dim>,
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		dim_param<DimIsPresent>,
		structure_param<T>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

	static_assert(DimMajor != DimMinor, "Cannot use the same name for two components of a dimension");
	static_assert(DimMinor != DimIsPresent, "Cannot use the same name for two components of a dimension");
	static_assert(DimIsPresent != DimMajor, "Cannot use the same name for two components of a dimension");
	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>, "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>, "Dimension of this name already exists");
	static_assert(DimIsPresent == Dim || !T::signature::template any_accept<DimIsPresent>, "Dimension of this name already exists");
private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");
		using major_length = std::conditional_t<Original::arg_length::is_known, dynamic_arg_length, unknown_arg_length>;
		using minor_length = unknown_arg_length;
		using ispresent_length = dynamic_arg_length;
		using type = function_sig<DimMajor, major_length, function_sig<DimMinor, minor_length, function_sig<DimIsPresent, ispresent_length, typename Original::ret_sig>>>;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<length_in<DimIsPresent>>, "This dimension cannot be resized");
		const auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>, index_in<DimIsPresent>>();
		constexpr bool have_indices = State::template contains<index_in<DimMajor>, index_in<DimMinor>, index_in<DimIsPresent>>;
		if constexpr(State::template contains<length_in<DimMajor>, length_in<DimMinor>>) {
			const auto major_length = state.template get<length_in<DimMajor>>();
			const auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<length_in<Dim>, index_in<Dim>>(major_length*minor_length, major_index*minor_length + minor_index);
			} else {
				return clean_state.template with<length_in<Dim>>(major_length*minor_length);
			}
		} else if constexpr(State::template contains<length_in<DimMinor>>) {
			const auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<index_in<Dim>>(major_index*minor_length + minor_index);
			} else {
				return clean_state;
			}
		} else {
			return clean_state;
		}
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
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == DimIsPresent) {
			static_assert(State::template contains<length_in<DimMinor>>, "Length has not been set");
			static_assert(State::template contains<index_in<DimMajor>>, "Fix block index before querying this dimension (or pass the index in state)");
			static_assert(State::template contains<index_in<DimMinor>>, "Fix index within block before querying this dimension (or pass the index in state)");
			const auto major_index = state.template get<index_in<DimMajor>>();
			const auto minor_index = state.template get<index_in<DimMinor>>();
			const auto minor_length = state.template get<length_in<DimMinor>>();
			const auto full_length = sub_structure().template length<Dim>(sub_state(state));
			const auto full_index = major_index * minor_length + minor_index;
			return std::size_t(full_index < full_length);
		} else if constexpr(QDim == DimMinor) {
			static_assert(State::template contains<length_in<DimMinor>>, "Length has not been set");
			return state.template get<length_in<DimMinor>>();
		} else if constexpr(QDim == DimMajor) {
			if constexpr(State::template contains<length_in<DimMajor>>) {
				return state.template get<length_in<DimMajor>>();
			} else if constexpr(State::template contains<length_in<DimMinor>>) {
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto full_length = sub_structure().template length<Dim>(sub_state(state));
				return (full_length + minor_length - make_const<1>()) / minor_length;
			} else {
				static_assert(value_always_false<QDim>, "Length has not been set (and cannot be computed from the total size because block size has also not been set)");
			}
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
struct into_blocks_dynamic_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, Struct>(s); }
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
constexpr auto into_blocks_dynamic() {
	return into_blocks_dynamic_proto<Dim, DimMajor, DimMinor, DimIsPresent>();
}

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T, class MinorLenT>
struct into_blocks_static_t : strict_contain<T, MinorLenT> {
	using strict_contain<T, MinorLenT>::strict_contain;

	static constexpr char name[] = "into_blocks_static_t";
	using params = struct_params<
		dim_param<Dim>,
		dim_param<DimIsBorder>,
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		structure_param<T>,
		type_param<MinorLenT>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->template get<0>(); }

	[[nodiscard]]
	constexpr MinorLenT minor_length() const noexcept { return this->template get<1>(); }

	static_assert(DimIsBorder != DimMajor, "Cannot use the same name for multiple components of a dimension");
	static_assert(DimIsBorder != DimMinor, "Cannot use the same name for multiple components of a dimension");
	static_assert(DimMajor != DimMinor, "Cannot use the same name for multiple components of a dimension");
	static_assert(DimIsBorder == Dim || !T::signature::template any_accept<DimIsBorder>, "Dimension of this name already exists");
	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>, "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>, "Dimension of this name already exists");
private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");
		static_assert(Original::arg_length::is_known, "Length of the dimension to be split must be set before splitting");
		template<class, class>
		struct divmod {
			using quo = dynamic_arg_length;
			using rem = dynamic_arg_length;
		};
		template<std::size_t Num, std::size_t Denom>
		struct divmod<static_arg_length<Num>, static_arg_length<Denom>> {
			using quo = static_arg_length<Num / Denom>;
			using rem = static_arg_length<Num % Denom>;
		};
		using denom = arg_length_from_t<MinorLenT>;
		using dm = divmod<typename Original::arg_length, denom>;
		using body_type = function_sig<DimMajor, typename dm::quo, function_sig<DimMinor, denom, typename Original::ret_sig>>;
		using border_type = function_sig<DimMajor, static_arg_length<1>, function_sig<DimMinor, typename dm::rem, typename Original::ret_sig>>;
		using type = dep_function_sig<DimIsBorder, body_type, border_type>;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<length_in<DimIsBorder>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMajor>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMinor>>, "This dimension cannot be resized");
		const auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimIsBorder>, index_in<DimMajor>, index_in<DimMinor>>();
		if constexpr(State::template contains<index_in<DimIsBorder>, index_in<DimMajor>, index_in<DimMinor>>) {
			const auto minor_index = state.template get<index_in<DimMinor>>();
			if constexpr(is_body<State>()) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				return clean_state.template with<index_in<Dim>>(major_index*minor_length() + minor_index);
			} else /*border*/ {
				const auto major_length = sub_structure().template length<Dim>(clean_state) / minor_length();
				return clean_state.template with<index_in<Dim>>(major_length*minor_length() + minor_index);
			}
		} else {
			return clean_state;
		}
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
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == DimIsBorder) {
			static_assert(!State::template contains<length_in<DimIsBorder>>, "Cannot set length in this dimension");
			return make_const<2>();
		} else if constexpr(QDim == DimMinor) {
			if constexpr(is_body<State>()) {
				return minor_length();
			} else /*border*/ {
				return sub_structure().template length<Dim>(sub_state(state)) % minor_length();
			}
		} else if constexpr(QDim == DimMajor) {
			if constexpr(is_body<State>()) {
				return sub_structure().template length<Dim>(sub_state(state)) / minor_length();
			} else /*border*/ {
				return make_const<1>();
			}
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}

private:
	template<IsState State>
	[[nodiscard]]
	static constexpr bool is_body() noexcept {
		static_assert(State::template contains<index_in<DimIsBorder>>, "Index has not been set");
		constexpr auto is_border = state_get_t<State, index_in<DimIsBorder>>::value;
		static_assert(is_border == 0 || is_border == 1, "The is-border index must be set statically (use lit<0> or lit<1>)");
		return is_border == 0;
	}
};

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class MinorLenT>
struct into_blocks_static_proto : strict_contain<MinorLenT> {
	using strict_contain<MinorLenT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, Struct, MinorLenT>(s, this->get()); }
};

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class MinorLenT>
constexpr auto into_blocks_static(MinorLenT minor_length) {
	return into_blocks_static_proto<Dim, DimIsBorder, DimMajor, DimMinor, good_index_t<MinorLenT>>(minor_length);
}

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim, class T>
struct merge_blocks_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "merge_blocks_t";
	using params = struct_params<
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		dim_param<Dim>,
		structure_param<T>>;

	[[nodiscard]]
	constexpr T sub_structure() const noexcept { return this->get(); }

	static_assert(DimMajor != DimMinor, "Cannot merge a dimension with itself");
	static_assert(Dim == DimMajor || Dim == DimMinor || !T::signature::template any_accept<Dim>, "Dimension of this name already exists");
private:
	template<class Original>
	struct outer_dim_replacement {
		static_assert(!Original::dependent, "Cannot merge a tuple index");
		template<class OriginalInner>
		struct inner_dim_replacement {
			static_assert(!OriginalInner::dependent, "Cannot merge a tuple index");
			using OrigMajor = std::conditional_t<Original::dim == DimMajor, Original, OriginalInner>;
			using OrigMinor = std::conditional_t<Original::dim == DimMinor, Original, OriginalInner>;
			static_assert(OrigMajor::dim == DimMajor && OrigMinor::dim == DimMinor, "bug");
			static_assert(OrigMinor::arg_length::is_known, "The minor dimension length must be set before merging");

			template<bool = OrigMinor::arg_length::is_static && OrigMajor::arg_length::is_static, bool = OrigMajor::arg_length::is_known, class = void>
			struct merged_len;
			template<class Useless>
			struct merged_len<true, true, Useless> { using type = static_arg_length<OrigMajor::arg_length::value * OrigMinor::arg_length::value>; };
			template<class Useless>
			struct merged_len<false, true, Useless> { using type = dynamic_arg_length; };
			template<class Useless>
			struct merged_len<false, false, Useless> { using type = unknown_arg_length; };

			using type = function_sig<Dim, typename merged_len<>::type, typename OriginalInner::ret_sig>;
		};

		using type = typename Original::ret_sig::template replace<inner_dim_replacement, DimMinor, DimMajor>;
	};
public:
	using signature = typename T::signature::template replace<outer_dim_replacement, DimMajor, DimMinor>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		const auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, length_in<DimMajor>, index_in<DimMinor>, length_in<DimMinor>>();
		const auto minor_length = sub_structure().template length<DimMinor>(clean_state);
		if constexpr(State::template contains<length_in<Dim>>) {
			const auto dim_length = state.template get<length_in<Dim>>();
			const auto major_length = dim_length / minor_length;
			if constexpr(State::template contains<index_in<Dim>>) {
				const auto index = state.template get<index_in<Dim>>();
				return clean_state.template with<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>>(index / minor_length, index % minor_length, major_length);
			} else {
				return clean_state.template with<length_in<DimMajor>>(major_length);
			}
		} else {
			if constexpr(State::template contains<index_in<Dim>>) {
				const auto index = state.template get<index_in<Dim>>();
				return clean_state.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length, index % minor_length);
			} else {
				return clean_state;
			}
		}
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
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == Dim) {
			if constexpr(State::template contains<length_in<Dim>>) {
				return state.template get<length_in<Dim>>();
			} else {
				return sub_structure().template length<DimMajor>(sub_state(state)) * sub_structure().template length<DimMinor>(sub_state(state));
			}
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub>
	[[nodiscard]]
	constexpr auto strict_state_at(IsState auto state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim>
struct merge_blocks_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return merge_blocks_t<DimMajor, DimMinor, Dim, Struct>(s); }
};

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim = DimMinor>
constexpr auto merge_blocks() {
	return merge_blocks_proto<DimMajor, DimMinor, Dim>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_BLOCKS_HPP
