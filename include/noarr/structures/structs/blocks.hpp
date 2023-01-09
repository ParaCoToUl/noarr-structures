#ifndef NOARR_STRUCTURES_BLOCKS_HPP
#define NOARR_STRUCTURES_BLOCKS_HPP

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<char Dim, char DimMajor, char DimMinor, class T>
struct into_blocks_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static constexpr char name[] = "into_blocks_t";
	using params = struct_params<
		dim_param<Dim>,
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		structure_param<T>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	static_assert(DimMajor != DimMinor, "Cannot use the same name for both components of a dimension");
	static_assert(T::signature::template all_accept<Dim>, "The structure does not have a dimension of this name");
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

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr(Dim != DimMajor && Dim != DimMinor) {
			static_assert(!State::template contains<index_in<Dim>>, "Index in this dimension is overriden by a substructure");
			static_assert(!State::template contains<length_in<Dim>>, "Index in this dimension is overriden by a substructure");
		}
		auto clean_state = state.template remove<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>();
		constexpr bool have_indices = State::template contains<index_in<DimMajor>> && State::template contains<index_in<DimMinor>>;
		if constexpr(State::template contains<length_in<DimMajor>> && State::template contains<length_in<DimMinor>>) {
			auto major_length = state.template get<length_in<DimMajor>>();
			auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				auto major_index = state.template get<index_in<DimMajor>>();
				auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<length_in<Dim>, index_in<Dim>>(major_length*minor_length, major_index*minor_length + minor_index);
			} else {
				return clean_state.template with<length_in<Dim>>(major_length*minor_length);
			}
		} else if constexpr(State::template contains<length_in<DimMinor>>) {
			auto minor_length = state.template get<length_in<DimMinor>>();
			if constexpr(have_indices) {
				auto major_index = state.template get<index_in<DimMajor>>();
				auto minor_index = state.template get<index_in<DimMinor>>();
				return clean_state.template with<index_in<Dim>>(major_index*minor_length + minor_index);
			} else {
				return clean_state;
			}
		} else {
			return clean_state;
		}
	}

	template<class State>
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr auto strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<DimMajor>>, "Index has not been set");
		static_assert(State::template contains<index_in<DimMinor>>, "Index has not been set");
		static_assert(State::template contains<length_in<DimMinor>>, "Length has not been set");
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == DimMinor) {
			static_assert(State::template contains<length_in<DimMinor>>, "Length has not been set");
			// TODO check remaining state
			return state.template get<length_in<DimMinor>>();
		} else if constexpr(QDim == DimMajor) {
			if constexpr(State::template contains<length_in<DimMajor>>) {
				// TODO check remaining state
				return state.template get<length_in<DimMajor>>();
			} else if constexpr(State::template contains<length_in<DimMinor>>) {
				auto minor_length = state.template get<length_in<DimMinor>>();
				return sub_structure().template length<Dim>(sub_state(state)) / minor_length;
			} else {
				static_assert(value_always_false<QDim>, "Length has not been set (and cannot be computed from the total size because block size has also not been set)");
			}
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<char Dim, char DimMajor, char DimMinor>
struct into_blocks_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return into_blocks_t<Dim, DimMajor, DimMinor, Struct>(s); }
};

template<char Dim, char DimMajor, char DimMinor>
constexpr auto into_blocks() {
	return into_blocks_proto<Dim, DimMajor, DimMinor>();
}

template<char Dim, char DimIsBorder, char DimMajor, char DimMinor, class T, class MinorLenT>
struct into_blocks_border_t : contain<T, MinorLenT> {
	using base = contain<T, MinorLenT>;
	using base::base;

	static constexpr char name[] = "into_blocks_border_t";
	using params = struct_params<
		dim_param<Dim>,
		dim_param<DimIsBorder>,
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		structure_param<T>,
		type_param<MinorLenT>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }
	constexpr MinorLenT minor_length() const noexcept { return base::template get<1>(); }

	static_assert(DimIsBorder != DimMajor, "Cannot use the same name for multiple components of a dimension");
	static_assert(DimIsBorder != DimMinor, "Cannot use the same name for multiple components of a dimension");
	static_assert(DimMajor != DimMinor, "Cannot use the same name for multiple components of a dimension");
	static_assert(T::signature::template all_accept<Dim>, "The structure does not have a dimension of this name");
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
		using border_type = function_sig<DimMinor, typename dm::rem, typename Original::ret_sig>;
		using type = dep_function_sig<DimIsBorder, body_type, border_type>;
	};
public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<length_in<DimIsBorder>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMajor>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMinor>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<Dim>>, "Index in this dimension is overriden by a substructure");
		if constexpr(Dim != DimIsBorder && Dim != DimMajor && Dim != DimMinor) {
			static_assert(!State::template contains<index_in<Dim>>, "Index in this dimension is overriden by a substructure");
		}
		auto clean_state = state.template remove<index_in<DimIsBorder>, index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>, length_in<DimMinor>>();
		if constexpr(State::template contains<index_in<DimIsBorder>> && State::template contains<index_in<DimMinor>>) {
			auto minor_index = state.template get<index_in<DimMinor>>();
			if constexpr(is_body<State>()) {
				if constexpr(State::template contains<index_in<DimMajor>>) {
					auto major_index = state.template get<index_in<DimMajor>>();
					return clean_state.template with<index_in<Dim>>(major_index*minor_length() + minor_index);
				} else {
					return clean_state;
				}
			} else /*border*/ {
				if constexpr(true) { // no other requirements, we do not need major
					auto major_length = sub_structure().template length<Dim>(clean_state) / minor_length();
					return clean_state.template with<index_in<Dim>>(major_length*minor_length() + minor_index);
				} else {
					return clean_state;
				}
			}
		} else {
			return clean_state;
		}
	}

	template<class State>
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr auto strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<DimIsBorder>>, "Index has not been set");
		static_assert(!is_body<State>() || State::template contains<index_in<DimMajor>>, "Index has not been set");
		static_assert(State::template contains<index_in<DimMinor>>, "Index has not been set");
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == DimIsBorder) {
			static_assert(!State::template contains<length_in<DimIsBorder>>, "Cannot set length in this dimension");
			// TODO check remaining state
			return make_const<2>();
		} else if constexpr(QDim == DimMinor) {
			// TODO check remaining state
			if constexpr(is_body<State>()) {
				return minor_length();
			} else /*border*/ {
				return sub_structure().template length<Dim>(sub_state(state)) % minor_length();
			}
		} else if constexpr(QDim == DimMajor) {
			// TODO check remaining state
			static_assert(is_body<State>(), "This dimension is not valid in the border");
			return sub_structure().template length<Dim>(sub_state(state)) / minor_length();
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, class State>
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}

private:
	template<class State>
	static constexpr bool is_body() noexcept {
		static_assert(State::template contains<index_in<DimIsBorder>>, "Index has not been set");
		using is_border = state_get_t<State, index_in<DimIsBorder>>;
		static_assert(is_border::value == 0 || is_border::value == 1, "The is-border index must be set statically (use idx<0> or idx<1>)");
		return is_border::value == 0;
	}
};

template<char Dim, char DimIsBorder, char DimMajor, char DimMinor, class MinorLenT>
struct into_blocks_border_proto : contain<MinorLenT> {
	using base = contain<MinorLenT>;
	using base::base;

	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return into_blocks_border_t<Dim, DimIsBorder, DimMajor, DimMinor, Struct, MinorLenT>(s, base::template get<0>()); }
};

template<char Dim, char DimIsBorder, char DimMajor, char DimMinor, class MinorLenT>
constexpr auto into_blocks(MinorLenT minor_length) {
	return into_blocks_border_proto<Dim, DimIsBorder, DimMajor, DimMinor, good_index_t<MinorLenT>>(minor_length);
}

template<char DimMajor, char DimMinor, char Dim, class T>
struct merge_blocks_t : contain<T> {
	using base = contain<T>;
	using base::base;

	static constexpr char name[] = "merge_blocks_t";
	using params = struct_params<
		dim_param<DimMajor>,
		dim_param<DimMinor>,
		dim_param<Dim>,
		structure_param<T>>;

	constexpr T sub_structure() const noexcept { return base::template get<0>(); }

	static_assert(DimMajor != DimMinor, "Cannot merge a dimension with itself");
	static_assert(T::signature::template all_accept<DimMajor>, "The structure does not have a dimension of this name");
	static_assert(T::signature::template all_accept<DimMinor>, "The structure does not have a dimension of this name");
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

	template<class State>
	constexpr auto sub_state(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr(DimMajor != Dim) {
			static_assert(!State::template contains<index_in<DimMajor>>, "Index in this dimension is overriden by a substructure");
			static_assert(!State::template contains<length_in<DimMajor>>, "Index in this dimension is overriden by a substructure");
		}
		if constexpr(DimMinor != Dim) {
			static_assert(!State::template contains<index_in<DimMinor>>, "Index in this dimension is overriden by a substructure");
			static_assert(!State::template contains<length_in<DimMinor>>, "Index in this dimension is overriden by a substructure");
		}
		auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>>();
		auto minor_length = sub_structure().template length<DimMinor>(clean_state);
		if constexpr(State::template contains<length_in<Dim>>) {
			auto length = state.template get<length_in<Dim>>();
			auto major_length = length / minor_length;
			if constexpr(State::template contains<index_in<Dim>>) {
				auto index = state.template get<index_in<Dim>>();
				return clean_state.template with<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>>(index / minor_length, index % minor_length, major_length);
			} else {
				return clean_state.template with<length_in<DimMajor>>(major_length);
			}
		} else {
			if constexpr(State::template contains<index_in<Dim>>) {
				auto index = state.template get<index_in<Dim>>();
				return clean_state.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length, index % minor_length);
			} else {
				return clean_state;
			}
		}
	}

	template<class State>
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<class Sub, class State>
	constexpr auto strict_offset_of(State state) const noexcept {
		static_assert(State::template contains<index_in<Dim>>, "Index has not been set");
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<char QDim, class State>
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>, "This dimension is already fixed, it cannot be used from outside");
		if constexpr(QDim == Dim) {
			if constexpr(State::template contains<length_in<Dim>>) {
				// TODO check remaining state
				return state.template get<length_in<Dim>>();
			} else {
				auto clean_state = state.template remove<index_in<Dim>, length_in<Dim>>();
				return sub_structure().template length<DimMajor>(clean_state) * sub_structure().template length<DimMinor>(clean_state);
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

template<char DimMajor, char DimMinor, char Dim>
struct merge_blocks_proto {
	static constexpr bool is_proto_struct = true;

	template<class Struct>
	constexpr auto instantiate_and_construct(Struct s) const noexcept { return merge_blocks_t<DimMajor, DimMinor, Dim, Struct>(s); }
};

template<char DimMajor, char DimMinor, char Dim>
constexpr auto merge_blocks() {
	return merge_blocks_proto<DimMajor, DimMinor, Dim>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_BLOCKS_HPP
