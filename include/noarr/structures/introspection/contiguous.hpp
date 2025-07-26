#ifndef NOARR_STRUCTURES_CONTIGUOUS_HPP
#define NOARR_STRUCTURES_CONTIGUOUS_HPP

#include <cstddef>

#include "../base/state.hpp"

#include "../structs/bcast.hpp"
#include "../structs/layouts.hpp"
#include "../structs/scalar.hpp"
#include "../structs/setters.hpp"
#include "../structs/views.hpp"

#include "../structs/slice.hpp"

#include "../structs/blocks.hpp"
#include "../structs/zcurve.hpp"

namespace noarr {

namespace helpers {

template<class T, IsState State>
struct is_contiguous : std::false_type {};

template<class ValueType, IsState State>
struct is_contiguous<scalar<ValueType>, State> : std::true_type {};

template<IsDim auto Dim, class T, IsState State>
struct is_contiguous<vector_t<Dim, T>, State> {
private:
	using Structure = vector_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		if constexpr (state_contains<State, length_in<Dim>>) {
			return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class... Ts, IsState State>
struct is_contiguous<tuple_t<Dim, Ts...>, State> {
private:
	using Structure = tuple_t<Dim, Ts...>;

	static constexpr bool get_value() noexcept {
		if constexpr (state_contains<State, index_in<Dim>>) {
			constexpr std::size_t index = state_get_t<State, index_in<Dim>>::value;

			using sub_struct = typename Structure::template sub_structure_t<index>;
			using sub_state = typename Structure::template sub_state_t<State>;

			return is_contiguous<sub_struct, sub_state>::value;
		} else {
			return (... && is_contiguous<Ts, typename Structure::template sub_state_t<State>>::value);
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, IsState State>
struct is_contiguous<bcast_t<Dim, T>, State> {
private:
	using Structure = bcast_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		if constexpr (state_contains<State, length_in<Dim>>) {
			return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class IdxT, IsState State>
struct is_contiguous<fix_t<Dim, T, IdxT>, State> {
private:
	using Structure = fix_t<Dim, T, IdxT>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class LenT, IsState State>
struct is_contiguous<set_length_t<Dim, T, LenT>, State> {
private:
	using Structure = set_length_t<Dim, T, LenT>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<class T, auto... Dims, IsState State>
requires IsDimPack<decltype(Dims)...>
struct is_contiguous<reorder_t<T, Dims...>, State> {
private:
	using Structure = reorder_t<T, Dims...>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, IsState State>
struct is_contiguous<hoist_t<Dim, T>, State> {
private:
	using Structure = hoist_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<class T, auto... DimPairs, class State>
requires IsDimPack<decltype(DimPairs)...> && (sizeof...(DimPairs) % 2 == 0) && IsState<State>
struct is_contiguous<rename_t<T, DimPairs...>, State> {
private:
	using Structure = rename_t<T, DimPairs...>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<class T, auto DimA, auto DimB, auto Dim, class State>
requires IsDim<decltype(DimA)> && IsDim<decltype(DimB)> && IsDim<decltype(Dim)> && (DimA != DimB) && IsState<State>
struct is_contiguous<join_t<T, DimA, DimB, Dim>, State> {
private:
	using Structure = join_t<T, DimA, DimB, Dim>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class StartT, IsState State>
struct is_contiguous<shift_t<Dim, T, StartT>, State> {
private:
	using Structure = shift_t<Dim, T, StartT>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class StartT, class LenT, IsState State>
struct is_contiguous<slice_t<Dim, T, StartT, LenT>, State> {
private:
	using Structure = slice_t<Dim, T, StartT, LenT>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class StartT, class EndT, IsState State>
struct is_contiguous<span_t<Dim, T, StartT, EndT>, State> {
private:
	using Structure = span_t<Dim, T, StartT, EndT>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename slice_t<Dim, T, StartT, EndT>::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, class StartT, class StrideT, IsState State>
struct is_contiguous<step_t<Dim, T, StartT, StrideT>, State> {
private:
	using Structure = step_t<Dim, T, StartT, StrideT>;

	static constexpr bool get_value() noexcept {
		if constexpr (state_contains<State, index_in<Dim>>) {
			return is_contiguous<T, typename step_t<Dim, T, StartT, StrideT>::template sub_state_t<State>>::value;
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, class T, IsState State>
struct is_contiguous<reverse_t<Dim, T>, State> {
private:
	using Structure = reverse_t<Dim, T>;

	static constexpr bool get_value() noexcept {
		return is_contiguous<T, typename reverse_t<Dim, T>::template sub_state_t<State>>::value;
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, class T, IsState State>
requires (DimMajor != DimMinor)
struct is_contiguous<into_blocks_t<Dim, DimMajor, DimMinor, T>, State> {
private:
	using Structure = into_blocks_t<Dim, DimMajor, DimMinor, T>;
	using indexless_state = decltype(std::declval<State>().template remove<index_in<DimMajor>, index_in<DimMinor>>());

	static constexpr bool get_value() noexcept {
		if constexpr (Structure::template has_length<DimMajor, indexless_state>() &&
		              Structure::template has_length<DimMinor, indexless_state>()) {
			if constexpr (state_contains<State, index_in<DimMinor>> &&
			              !state_contains<State, index_in<DimMajor>>) {
				return false;
			} else {
				return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
			}
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T, IsState State>
requires (DimMajor != DimMinor) && (DimMinor != DimIsPresent) && (DimIsPresent != DimMajor)
struct is_contiguous<into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>, State> {
private:
	using Structure = into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, T>;
	using indexless_state = decltype(std::declval<State>().template remove<index_in<DimMajor>, index_in<DimMinor>>());

	static constexpr bool get_value() noexcept {
		if constexpr (Structure::template has_length<DimMajor, indexless_state>() &&
		              Structure::template has_length<DimMinor, indexless_state>()) {
			if constexpr (state_contains<State, index_in<DimMinor>> &&
			              !state_contains<State, index_in<DimMajor>>) {
				return false;
			} else {
				return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
			}
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T, class MinorLenT,
         IsState State>
requires (DimIsBorder != DimMajor) && (DimIsBorder != DimMinor) && (DimMajor != DimMinor)
struct is_contiguous<into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>, State> {
private:
	using Structure = into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, T, MinorLenT>;
	using indexless_state =
		decltype(std::declval<State>()
	                 .template remove<index_in<DimMajor>, index_in<DimMinor>, index_in<DimIsBorder>>());

	static constexpr bool get_value() noexcept {
		if constexpr (state_contains<State, index_in<DimIsBorder>>) {
			if constexpr (Structure::template has_length<DimMajor, indexless_state>() &&
			              Structure::template has_length<DimMinor, indexless_state>()) {
				if constexpr (state_contains<State, index_in<DimMinor>> &&
				              !state_contains<State, index_in<DimMajor>>) {
					return false;
				} else {
					return is_contiguous<T, typename Structure::template sub_state_t<State>>::value;
				}
			} else {
				return false;
			}
		} else {
			return false;
		}
	}

public:
	using value_type = bool;
	static constexpr bool value = get_value();
};

} // namespace helpers

template<class T, class State>
concept IsContiguous = requires {
	requires IsStruct<T>;
	requires IsState<State>;

	requires helpers::is_contiguous<T, State>::value;
};

template<class T, IsState State>
constexpr bool is_contiguous() noexcept {
	return helpers::is_contiguous<T, State>::value;
}

template<IsState State = state<>>
constexpr auto is_contiguous(State /*unused*/) noexcept {
	return []<class Struct>(Struct /*unused*/) constexpr noexcept { return is_contiguous<Struct, State>(); };
}

template<class T, IsState State>
constexpr bool is_contiguous(const T & /*unused*/, State /*unused*/) noexcept {
	return is_contiguous<T, State>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_CONTIGUOUS_HPP
