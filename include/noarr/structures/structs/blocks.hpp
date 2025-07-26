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
requires (DimMajor != DimMinor)
struct into_blocks_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "into_blocks_t";
	using params = struct_params<dim_param<Dim>, dim_param<DimMajor>, dim_param<DimMinor>, structure_param<T>>;

	template<IsState State>
	[[nodiscard]]
	constexpr T sub_structure(State /*state*/) const noexcept {
		return this->get();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return this->get();
	}

	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>,
	              "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>,
	              "Dimension of this name already exists");

	template<IsState State>
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, index_in<DimMinor>,
		                             length_in<DimMajor>, length_in<DimMinor>>();
	}

	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");
		using major_length = dynamic_arg_length;
		using minor_length = dynamic_arg_length;
		using type =
			function_sig<DimMajor, major_length, function_sig<DimMinor, minor_length, typename Original::ret_sig>>;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, T sub_structure) noexcept {
			using namespace constexpr_arithmetic;
			constexpr bool have_indices = State::template contains<index_in<DimMajor>, index_in<DimMinor>>;
			if constexpr (State::template contains<length_in<DimMajor>, length_in<DimMinor>>) {
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto minor_length = state.template get<length_in<DimMinor>>();
				if constexpr (have_indices) {
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return clean_state(state).template with<length_in<Dim>, index_in<Dim>>(
						major_length * minor_length, major_index * minor_length + minor_index);
				} else {
					return clean_state(state).template with<length_in<Dim>>(major_length * minor_length);
				}
			} else if constexpr (State::template contains<length_in<DimMinor>>) {
				if constexpr (have_indices) {
					const auto minor_length = state.template get<length_in<DimMinor>>();
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return clean_state(state).template with<index_in<Dim>>(major_index * minor_length + minor_index);
				} else {
					return clean_state(state);
				}
			} else if constexpr (!sub_structure_t::template has_length<Dim, clean_state_t<State>>()) {
				return clean_state(state);
			} else if constexpr (State::template contains<length_in<DimMajor>>) {
				if constexpr (have_indices) {
					const auto cs = clean_state(state);
					const auto major_length = state.template get<length_in<DimMajor>>();
					const auto minor_length = sub_structure.template length<Dim>(cs) / major_length;
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return cs.template with<index_in<Dim>>(major_index * minor_length + minor_index);
				} else {
					return clean_state(state);
				}
			} else {
				return clean_state(state);
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, sub_structure());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<T>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto align(State state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_offset_of<Sub, into_blocks_t, State>())
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (QDim == DimMinor) {
			if constexpr (State::template contains<index_in<DimMinor>>) {
				return false;
			} else if constexpr (State::template contains<length_in<DimMinor>>) {
				static_assert(!State::template contains<length_in<DimMajor>> ||
				                  !sub_structure_t::template has_length<Dim, clean_state_t<State>>(),
				              "Two different ways to determine the length of the minor dimension");
				return true;
			} else if constexpr (State::template contains<length_in<DimMajor>> &&
			                     sub_structure_t::template has_length<Dim, sub_state_t<State>>()) {
				return true;
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (State::template contains<index_in<DimMajor>>) {
				return false;
			} else if constexpr (State::template contains<length_in<DimMajor>>) {
				static_assert(!State::template contains<length_in<DimMinor>> ||
				                  !sub_structure_t::template has_length<Dim, clean_state_t<State>>(),
				              "Two different ways to determine the length of the major dimension");
				return true;
			} else if constexpr (State::template contains<length_in<DimMinor>> &&
			                     sub_structure_t::template has_length<Dim, sub_state_t<State>>()) {
				return true;
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			// This dimension is consumed by into_blocks
			return false;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		static_assert(!State::template contains<index_in<QDim>>,
		              "This dimension is already fixed, it cannot be used from outside");
		if constexpr (QDim == DimMinor) {
			if constexpr (State::template contains<length_in<DimMinor>>) {
				return state.template get<length_in<DimMinor>>();
			} else {
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto full_length = sub_structure().template length<Dim>(clean_state(state));
				return full_length / major_length;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (State::template contains<length_in<DimMajor>>) {
				return state.template get<length_in<DimMajor>>();
			} else {
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto full_length = sub_structure().template length<Dim>(sub_state(state));
				return full_length / minor_length;
			}
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_state_at<Sub, into_blocks_t, State>())
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor>
struct into_blocks_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return into_blocks_t<Dim, DimMajor, DimMinor, Struct>(s);
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor = Dim>
constexpr auto into_blocks() {
	return into_blocks_proto<Dim, DimMajor, DimMinor>();
}

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent, class T>
requires (DimMajor != DimMinor) && (DimMinor != DimIsPresent) && (DimIsPresent != DimMajor)
struct into_blocks_dynamic_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "into_blocks_dynamic_t";
	using params = struct_params<dim_param<Dim>, dim_param<DimMajor>, dim_param<DimMinor>, dim_param<DimIsPresent>,
	                             structure_param<T>>;

	template<IsState State>
	[[nodiscard]]
	constexpr T sub_structure(State /*state*/) const noexcept {
		return this->get();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return this->get();
	}

	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>,
	              "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>,
	              "Dimension of this name already exists");
	static_assert(DimIsPresent == Dim || !T::signature::template any_accept<DimIsPresent>,
	              "Dimension of this name already exists");

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		static_assert(!State::template contains<length_in<DimIsPresent>>, "This dimension cannot be resized");
		return state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, index_in<DimMinor>,
		                             length_in<DimMajor>, length_in<DimMinor>, index_in<DimIsPresent>>();
	}

	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");
		using major_length = dynamic_arg_length;
		using minor_length = dynamic_arg_length;
		using ispresent_length = dynamic_arg_length;
		using type =
			function_sig<DimMajor, major_length,
		                 function_sig<DimMinor, minor_length,
		                              function_sig<DimIsPresent, ispresent_length, typename Original::ret_sig>>>;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, T sub_structure) noexcept {
			using namespace constexpr_arithmetic;
			constexpr bool have_indices = State::template contains<index_in<DimMajor>, index_in<DimMinor>>;
			if constexpr (State::template contains<length_in<DimMajor>, length_in<DimMinor>>) {
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto minor_length = state.template get<length_in<DimMinor>>();
				if constexpr (have_indices) {
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return clean_state(state).template with<length_in<Dim>, index_in<Dim>>(
						major_length * minor_length, major_index * minor_length + minor_index);
				} else {
					return clean_state(state).template with<length_in<Dim>>(major_length * minor_length);
				}
			} else if constexpr (State::template contains<length_in<DimMinor>>) {
				if constexpr (have_indices) {
					const auto minor_length = state.template get<length_in<DimMinor>>();
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return clean_state(state).template with<index_in<Dim>>(major_index * minor_length + minor_index);
				} else {
					return clean_state(state);
				}
			} else if constexpr (State::template contains<length_in<DimMajor>>) {
				if constexpr (have_indices) {
					const auto cs = clean_state(state);
					const auto major_length = state.template get<length_in<DimMajor>>();
					const auto minor_length =
						(sub_structure.template length<Dim>(cs) + major_length - make_const<1>()) / major_length;
					const auto major_index = state.template get<index_in<DimMajor>>();
					const auto minor_index = state.template get<index_in<DimMinor>>();
					return cs.template with<index_in<Dim>>(major_index * minor_length + minor_index);
				} else {
					return clean_state(state);
				}
			} else {
				return clean_state(state);
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, sub_structure());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<T>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto align(State state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		if constexpr (!State::template contains<index_in<DimMajor>, index_in<DimMinor>, index_in<DimIsPresent>>) {
			return false;
		} else {
			return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
		}
	}

	template<class Sub, IsState State>
	requires (has_offset_of<Sub, into_blocks_dynamic_t, State>())
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (QDim == DimIsPresent) {
			if constexpr (!State::template contains<index_in<DimMajor>, index_in<DimMinor>>) {
				return false;
			} else {
				if constexpr (!State::template contains<length_in<DimMinor>> &&
				              !State::template contains<length_in<DimMajor>>) {
					return false;
				} else if constexpr (State::template contains<length_in<DimMinor>> &&
				                     State::template contains<length_in<DimMajor>>) {
					static_assert(!sub_structure_t::template has_length<Dim, clean_state_t<State>>(),
					              "Two different ways to determine the length of the minor dimension");
					return true;
				} else {
					return sub_structure_t::template has_length<Dim, clean_state_t<State>>();
				}
			}
		} else if constexpr (QDim == DimMinor) {
			if constexpr (State::template contains<index_in<DimMinor>>) {
				return false;
			} else if constexpr (State::template contains<length_in<DimMinor>>) {
				static_assert(!State::template contains<length_in<DimMajor>> ||
				                  !sub_structure_t::template has_length<Dim, clean_state_t<State>>(),
				              "Two different ways to determine the length of the minor dimension");
				return true;
			} else if constexpr (State::template contains<length_in<DimMajor>> &&
			                     sub_structure_t::template has_length<Dim, sub_state_t<State>>()) {
				return true;
			} else {
				return false;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (State::template contains<index_in<DimMajor>>) {
				return false;
			} else if constexpr (State::template contains<length_in<DimMajor>>) {
				static_assert(!State::template contains<length_in<DimMinor>> ||
				                  !sub_structure_t::template has_length<Dim, clean_state_t<State>>(),
				              "Two different ways to determine the length of the major dimension");
				return true;
			} else if constexpr (State::template contains<length_in<DimMinor>> &&
			                     sub_structure_t::template has_length<Dim, sub_state_t<State>>()) {
				return true;
			} else {
				return false;
			}
		} else if constexpr (QDim == Dim) {
			// This dimension is consumed by into_blocks_dynamic
			return false;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimIsPresent) {
			if constexpr (State::template contains<length_in<DimMinor>, length_in<DimMajor>>) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto full_length = minor_length * major_length;
				const auto full_index = major_index * minor_length + minor_index;
				return std::size_t(full_index < full_length);
			} else if constexpr (State::template contains<length_in<DimMinor>>) {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto full_length = sub_structure().template length<Dim>(clean_state(state));
				const auto full_index = major_index * minor_length + minor_index;
				return std::size_t(full_index < full_length);
			} else {
				const auto major_index = state.template get<index_in<DimMajor>>();
				const auto minor_index = state.template get<index_in<DimMinor>>();
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto full_length = sub_structure().template length<Dim>(clean_state(state));
				const auto minor_length = (full_length + major_length - make_const<1>()) / major_length;
				const auto full_index = major_index * minor_length + minor_index;
				return std::size_t(full_index < full_length);
			}
		} else if constexpr (QDim == DimMinor) {
			if constexpr (State::template contains<length_in<DimMinor>>) {
				return state.template get<length_in<DimMinor>>();
			} else {
				const auto major_length = state.template get<length_in<DimMajor>>();
				const auto full_length = sub_structure().template length<Dim>(clean_state(state));
				return (full_length + major_length - make_const<1>()) / major_length;
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (State::template contains<length_in<DimMajor>>) {
				return state.template get<length_in<DimMajor>>();
			} else {
				const auto minor_length = state.template get<length_in<DimMinor>>();
				const auto full_length = sub_structure().template length<Dim>(sub_state(state));
				return (full_length + minor_length - make_const<1>()) / minor_length;
			}
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_state_at<Sub, into_blocks_dynamic_t, State>())
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
struct into_blocks_dynamic_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return into_blocks_dynamic_t<Dim, DimMajor, DimMinor, DimIsPresent, Struct>(s);
	}
};

template<IsDim auto Dim, IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto DimIsPresent>
constexpr auto into_blocks_dynamic() {
	return into_blocks_dynamic_proto<Dim, DimMajor, DimMinor, DimIsPresent>();
}

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class T, class MinorLenT>
requires (DimIsBorder != DimMajor) && (DimIsBorder != DimMinor) && (DimMajor != DimMinor)
struct into_blocks_static_t : strict_contain<T, MinorLenT> {
	using strict_contain<T, MinorLenT>::strict_contain;

	static constexpr char name[] = "into_blocks_static_t";
	using params = struct_params<dim_param<Dim>, dim_param<DimIsBorder>, dim_param<DimMajor>, dim_param<DimMinor>,
	                             structure_param<T>, type_param<MinorLenT>>;

	template<IsState State>
	[[nodiscard]]
	constexpr T sub_structure(State /*state*/) const noexcept {
		return this->template get<0>();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return this->template get<0>();
	}

	[[nodiscard]]
	constexpr MinorLenT minor_length() const noexcept {
		return this->template get<1>();
	}

	static_assert(DimIsBorder == Dim || !T::signature::template any_accept<DimIsBorder>,
	              "Dimension of this name already exists");
	static_assert(DimMajor == Dim || !T::signature::template any_accept<DimMajor>,
	              "Dimension of this name already exists");
	static_assert(DimMinor == Dim || !T::signature::template any_accept<DimMinor>,
	              "Dimension of this name already exists");

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		static_assert(!State::template contains<length_in<DimIsBorder>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMajor>>, "This dimension cannot be resized");
		static_assert(!State::template contains<length_in<DimMinor>>, "This dimension cannot be resized");
		return state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimIsBorder>, index_in<DimMajor>,
		                             index_in<DimMinor>>();
	}

	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

private:
	template<class Original>
	struct dim_replacement {
		static_assert(!Original::dependent, "Cannot split a tuple index into blocks");

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
		using body_type =
			function_sig<DimMajor, typename dm::quo, function_sig<DimMinor, denom, typename Original::ret_sig>>;
		using border_type = function_sig<DimMajor, static_arg_length<1>,
		                                 function_sig<DimMinor, typename dm::rem, typename Original::ret_sig>>;
		using type = dep_function_sig<DimIsBorder, body_type, border_type>;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, T sub_structure, MinorLenT minor_length) noexcept {
			using namespace constexpr_arithmetic;
			if constexpr (State::template contains<index_in<DimIsBorder>, index_in<DimMajor>, index_in<DimMinor>>) {
				const auto minor_index = state.template get<index_in<DimMinor>>();
				if constexpr (is_body<State>()) {
					const auto major_index = state.template get<index_in<DimMajor>>();
					return clean_state(state).template with<index_in<Dim>>(major_index * minor_length + minor_index);
				} else /*border*/ {
					const auto cs = clean_state(state);
					const auto major_length = sub_structure.template length<Dim>(cs) / minor_length;
					return cs.template with<index_in<Dim>>(major_length * minor_length + minor_index);
				}
			} else {
				return clean_state(state);
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, sub_structure(), minor_length());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t =
		decltype(impl::sub_state(std::declval<State>(), std::declval<sub_structure_t>(), std::declval<MinorLenT>()));

	template<IsState State>
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto align(State state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_offset_of<Sub, into_blocks_static_t, State>())
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (State::template contains<index_in<QDim>>) {
			// This dimension is already fixed, it cannot be used from outside
			return false;
		} else if constexpr (QDim == DimMajor) {
			if constexpr (is_body<State>()) {
				return sub_structure_t::template has_length<Dim, sub_state_t<State>>();
			} else /*border*/ {
				return true;
			}
		} else if constexpr (QDim == DimMinor) {
			if constexpr (is_body<State>()) {
				return true;
			} else /*border*/ {
				return sub_structure_t::template has_length<Dim, sub_state_t<State>>();
			}
		} else if constexpr (QDim == DimIsBorder) {
			static_assert(!State::template contains<length_in<DimIsBorder>>, "Cannot set length in this dimension");
			return true;
		} else if constexpr (QDim == Dim) {
			// This dimension is consumed by into_blocks_static
			return false;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr (QDim == DimIsBorder) {
			return make_const<2>();
		} else if constexpr (QDim == DimMinor) {
			if constexpr (is_body<State>()) {
				return minor_length();
			} else /*border*/ {
				return sub_structure().template length<Dim>(sub_state(state)) % minor_length();
			}
		} else if constexpr (QDim == DimMajor) {
			if constexpr (is_body<State>()) {
				return sub_structure().template length<Dim>(sub_state(state)) / minor_length();
			} else /*border*/ {
				return make_const<1>();
			}
		} else {
			static_assert(QDim != Dim, "Index in this dimension is overriden by a substructure");
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_state_at<Sub, into_blocks_static_t, State>())
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}

private:
	template<IsState State>
	[[nodiscard]]
	static constexpr bool is_body() noexcept {
		static_assert(State::template contains<index_in<DimIsBorder>>, "Index has not been set");
		constexpr auto is_border = state_get_t<State, index_in<DimIsBorder>>::value;
		static_assert(is_border == 0 || is_border == 1,
		              "The is-border index must be set statically (use lit<0> or lit<1>)");
		return is_border == 0;
	}
};

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class MinorLenT>
struct into_blocks_static_proto : strict_contain<MinorLenT> {
	using strict_contain<MinorLenT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return into_blocks_static_t<Dim, DimIsBorder, DimMajor, DimMinor, Struct, MinorLenT>(s, this->get());
	}
};

template<IsDim auto Dim, IsDim auto DimIsBorder, IsDim auto DimMajor, IsDim auto DimMinor, class MinorLenT>
constexpr auto into_blocks_static(MinorLenT minor_length) {
	return into_blocks_static_proto<Dim, DimIsBorder, DimMajor, DimMinor, good_index_t<MinorLenT>>(minor_length);
}

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim, class T>
requires (DimMajor != DimMinor)
struct merge_blocks_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "merge_blocks_t";
	using params = struct_params<dim_param<DimMajor>, dim_param<DimMinor>, dim_param<Dim>, structure_param<T>>;

	template<IsState State>
	[[nodiscard]]
	constexpr T sub_structure(State /*state*/) const noexcept {
		return this->get();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return this->get();
	}

	static_assert(Dim == DimMajor || Dim == DimMinor || !T::signature::template any_accept<Dim>,
	              "Dimension of this name already exists");

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<index_in<Dim>, length_in<Dim>, index_in<DimMajor>, length_in<DimMajor>,
		                             index_in<DimMinor>, length_in<DimMinor>>();
	}

	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

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

			template<bool = OrigMinor::arg_length::is_static && OrigMajor::arg_length::is_static, class = void>
			struct merged_len;

			template<class Useless>
			struct merged_len<true, Useless> {
				using type = static_arg_length<OrigMajor::arg_length::value * OrigMinor::arg_length::value>;
			};

			template<class Useless>
			struct merged_len<false, Useless> {
				using type = dynamic_arg_length;
			};

			using type = function_sig<Dim, typename merged_len<>::type, typename OriginalInner::ret_sig>;
		};

		using type = typename Original::ret_sig::template replace<inner_dim_replacement, DimMinor, DimMajor>;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, T sub_structure) noexcept {
			using namespace constexpr_arithmetic;
			const auto cs = clean_state(state);
			if constexpr (sub_structure_t::template has_length<DimMinor, clean_state_t<State>>() &&
			              sub_structure_t::template has_length<DimMajor, clean_state_t<State>>()) {
				if constexpr (State::template contains<index_in<Dim>>) {
					const auto minor_length = sub_structure.template length<DimMinor>(cs);
					const auto index = state.template get<index_in<Dim>>();
					return cs.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length,
					                                                                index % minor_length);
				} else {
					return cs;
				}
			} else if constexpr (sub_structure_t::template has_length<DimMinor, clean_state_t<State>>()) {
				const auto minor_length = sub_structure.template length<DimMinor>(cs);
				if constexpr (State::template contains<length_in<Dim>>) {
					const auto dim_length = state.template get<length_in<Dim>>();
					const auto major_length = dim_length / minor_length;
					if constexpr (State::template contains<index_in<Dim>>) {
						const auto index = state.template get<index_in<Dim>>();
						return cs.template with<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMajor>>(
							index / minor_length, index % minor_length, major_length);
					} else {
						return cs.template with<length_in<DimMajor>>(major_length);
					}
				} else {
					if constexpr (State::template contains<index_in<Dim>>) {
						const auto index = state.template get<index_in<Dim>>();
						return cs.template with<index_in<DimMajor>, index_in<DimMinor>>(index / minor_length,
						                                                                index % minor_length);
					} else {
						return cs;
					}
				}
			} else if constexpr (sub_structure_t::template has_length<DimMajor, clean_state_t<State>>()) {
				const auto major_length = sub_structure.template length<DimMajor>(cs);
				if constexpr (State::template contains<length_in<Dim>>) {
					const auto dim_length = state.template get<length_in<Dim>>();
					const auto minor_length = dim_length / major_length;
					if constexpr (State::template contains<index_in<Dim>>) {
						const auto index = state.template get<index_in<Dim>>();
						return cs.template with<index_in<DimMajor>, index_in<DimMinor>, length_in<DimMinor>>(
							index / minor_length, index % minor_length, minor_length);
					} else {
						return cs.template with<length_in<DimMinor>>(minor_length);
					}
				} else {
					return cs;
				}
			} else {
				return cs;
			}
		}
	};

public:
	using signature = typename T::signature::template replace<outer_dim_replacement, DimMajor, DimMinor>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, sub_structure());
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<sub_structure_t>()));

	template<IsState State>
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto size(State state) const noexcept {
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	requires (has_size<State>())
	[[nodiscard]]
	constexpr auto align(State state) const noexcept {
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_offset_of<Sub, merge_blocks_t, State>())
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept {
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		static_assert(!State::template contains<index_in<QDim>>,
		              "This dimension is already fixed, it cannot be used from outside");
		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>>) {
				static_assert(!sub_structure_t::template has_length<DimMajor, clean_state_t<State>>() ||
				                  !sub_structure_t::template has_length<DimMinor, clean_state_t<State>>(),
				              "Two different ways to determine the length of the major dimension");
				return true;
			} else {
				return sub_structure_t::template has_length<DimMajor, sub_state_t<State>>() &&
				       sub_structure_t::template has_length<DimMinor, sub_state_t<State>>();
			}
		} else if constexpr (QDim == DimMinor || QDim == DimMajor) {
			// This dimension is consumed by merge_blocks
			return false;
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires (has_length<QDim, State>())
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>>) {
				return state.template get<length_in<Dim>>();
			} else {
				return sub_structure().template length<DimMajor>(sub_state(state)) *
				       sub_structure().template length<DimMinor>(sub_state(state));
			}
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	requires (has_state_at<Sub, merge_blocks_t, State>())
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept {
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim>
struct merge_blocks_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return merge_blocks_t<DimMajor, DimMinor, Dim, Struct>(s);
	}
};

template<IsDim auto DimMajor, IsDim auto DimMinor, IsDim auto Dim = DimMinor>
constexpr auto merge_blocks() {
	return merge_blocks_proto<DimMajor, DimMinor, Dim>();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_BLOCKS_HPP
