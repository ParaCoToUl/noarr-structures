#ifndef NOARR_STRUCTURES_SLICE_HPP
#define NOARR_STRUCTURES_SLICE_HPP

#include <cstddef>
#include <type_traits>
#include <utility>

#include "../base/contain.hpp"
#include "../base/signature.hpp"
#include "../base/state.hpp"
#include "../base/structs_common.hpp"
#include "../base/utility.hpp"

namespace noarr {

template<IsDim auto Dim, class T, class StartT>
struct shift_t : strict_contain<T, StartT> {
	using strict_contain<T, StartT>::strict_contain;

	static constexpr char name[] = "shift_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>, type_param<StartT>>;

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
	constexpr StartT start() const noexcept {
		return this->template get<1>();
	}

private:
	template<class Original>
	struct dim_replacement;

	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		template<class L, class S>
		struct subtract {
			using type = dynamic_arg_length;
		};

		template<std::size_t L, std::size_t S>
		struct subtract<static_arg_length<L>, std::integral_constant<std::size_t, S>> {
			using type = static_arg_length<L - S>;
		};

		using type = function_sig<Dim, typename subtract<ArgLength, StartT>::type, RetSig>;
	};

	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(requires { StartT::value; }, "Cannot shift a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t len = sizeof...(RetSigs) - start;

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;

		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> {
			using type = dep_function_sig<Dim, typename original::template ret_sig<Indices + start>...>;
		};

		using type = typename pack_helper<>::type;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, StartT start) noexcept {
			using namespace constexpr_arithmetic;
			const auto tmp_state = clean_state(state);
			if constexpr (State::template contains<index_in<Dim>>) {
				if constexpr (State::template contains<length_in<Dim>>) {
					return tmp_state.template with<index_in<Dim>, length_in<Dim>>(
						state.template get<index_in<Dim>>() + start, state.template get<length_in<Dim>>() + start);
				} else {
					return tmp_state.template with<index_in<Dim>>(state.template get<index_in<Dim>>() + start);
				}
			} else {
				if constexpr (State::template contains<length_in<Dim>>) {
					return tmp_state.template with<length_in<Dim>>(state.template get<length_in<Dim>>() + start);
				} else {
					return tmp_state;
				}
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, start());
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<index_in<Dim>, length_in<Dim>>();
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<StartT>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, shift_t, State>())
	{
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>>) {
				return true;
			} else {
				return sub_structure_t::template has_length<Dim, sub_state_t<State>>();
			}
		} else {
			return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
		}
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			if constexpr (State::template contains<length_in<Dim>>) {
				return state.template get<length_in<Dim>>();
			} else {
				return sub_structure().template length<Dim>(sub_state(state)) - start();
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
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, shift_t, State>())
	{
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class StartT>
struct shift_proto : strict_contain<StartT> {
	using strict_contain<StartT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return shift_t<Dim, Struct, StartT>(s, this->get());
	}
};

/**
 * @brief shifts an index (or indices) given by dimension name(s) in a structure
 *
 * @tparam Dim: the dimension names
 * @param start: parameters for shifting the indices
 */
template<auto... Dims, class... StartT>
requires IsDimPack<decltype(Dims)...>
constexpr auto shift(StartT... start) noexcept {
	return (... ^ shift_proto<Dims, good_index_t<StartT>>(start));
}

template<>
constexpr auto shift<>() noexcept {
	return neutral_proto();
}

template<IsDim auto Dim, class T, class StartT, class LenT>
struct slice_t : strict_contain<T, StartT, LenT> {
	using strict_contain<T, StartT, LenT>::strict_contain;

	static constexpr char name[] = "slice_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>, type_param<StartT>, type_param<LenT>>;

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
	constexpr StartT start() const noexcept {
		return this->template get<1>();
	}

	[[nodiscard]]
	constexpr LenT len() const noexcept {
		return this->template get<2>();
	}

private:
	template<class Original>
	struct dim_replacement;

	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		using type = function_sig<Dim, arg_length_from_t<LenT>, RetSig>;
	};

	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(requires { StartT::value; }, "Cannot slice a tuple dimension dynamically");
		static_assert(requires { LenT::value; }, "Cannot slice a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t len = LenT::value;

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;

		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> {
			using type = dep_function_sig<Dim, typename original::template ret_sig<Indices + start>...>;
		};

		using type = typename pack_helper<>::type;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, StartT start) noexcept {
			using namespace constexpr_arithmetic;
			if constexpr (State::template contains<index_in<Dim>>) {
				return state.template with<index_in<Dim>>(state.template get<index_in<Dim>>() + start);
			} else {
				return state;
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, start());
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<length_in<Dim>, index_in<Dim>>();
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<StartT>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, slice_t, State>())
	{
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept
	requires (has_length<QDim, State>())
	{
		if constexpr (QDim == Dim) {
			return len();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, slice_t, State>())
	{
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set slice length");
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class StartT, class LenT>
struct slice_proto : strict_contain<StartT, LenT> {
	using strict_contain<StartT, LenT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return slice_t<Dim, Struct, StartT, LenT>(s, this->template get<0>(), this->template get<1>());
	}
};

template<IsDim auto Dim, class StartT, class LenT>
constexpr auto slice(StartT start, LenT len) noexcept {
	return slice_proto<Dim, good_index_t<StartT>, good_index_t<LenT>>(start, len);
}

template<IsDim auto Dim, class LenT>
constexpr auto slice(LenT len) noexcept {
	return slice_proto<Dim, good_index_t<lit_t<0>>, good_index_t<LenT>>(lit<0>, len);
}

template<IsDim auto Dim, class T, class StartT, class EndT>
struct span_t : strict_contain<T, StartT, EndT> {
	using strict_contain<T, StartT, EndT>::strict_contain;

	static constexpr char name[] = "span_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>, type_param<StartT>, type_param<EndT>>;

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
	constexpr StartT start() const noexcept {
		return this->template get<1>();
	}

	[[nodiscard]]
	constexpr EndT end() const noexcept {
		return this->template get<2>();
	}

private:
	template<class Original>
	struct dim_replacement;

	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		using type = function_sig<Dim, arg_length_from_t<EndT>, RetSig>;
	};

	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(requires { StartT::value; }, "Cannot span a tuple dimension dynamically");
		static_assert(requires { EndT::value; }, "Cannot span a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t end = EndT::value;

		template<class Indices = std::make_index_sequence<end - start>>
		struct pack_helper;

		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> {
			using type = dep_function_sig<Dim, typename original::template ret_sig<Indices + start>...>;
		};

		using type = typename pack_helper<>::type;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, StartT start) noexcept {
			using namespace constexpr_arithmetic;
			if constexpr (State::template contains<index_in<Dim>>) {
				return state.template with<index_in<Dim>>(state.template get<index_in<Dim>>() + start);
			} else {
				return state;
			}
		}
	};

public:
	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, start());
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<length_in<Dim>, index_in<Dim>>();
	}

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<StartT>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set span length");
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set span length");
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, span_t, State>())
	{
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set span length");
		return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept
	requires (has_length<QDim, State>())
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			return end() - start();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set span length");
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, span_t, State>())
	{
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class StartT, class EndT>
struct span_proto : strict_contain<StartT, EndT> {
	using strict_contain<StartT, EndT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return span_t<Dim, Struct, StartT, EndT>(s, this->template get<0>(), this->template get<1>());
	}
};

template<IsDim auto Dim, class StartT, class EndT>
constexpr auto span(StartT start, EndT end) noexcept {
	return span_proto<Dim, good_index_t<StartT>, good_index_t<EndT>>(start, end);
}

template<IsDim auto Dim, class EndT>
constexpr auto span(EndT end) noexcept {
	return span_proto<Dim, good_index_t<lit_t<0>>, good_index_t<EndT>>(lit<0>, end);
}

template<IsDim auto Dim, class T, class StartT, class StrideT>
struct step_t : strict_contain<T, StartT, StrideT> {
	using strict_contain<T, StartT, StrideT>::strict_contain;

	static constexpr char name[] = "step_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>, type_param<StartT>, type_param<StrideT>>;

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
	constexpr StartT start() const noexcept {
		return this->template get<1>();
	}

	[[nodiscard]]
	constexpr StrideT stride() const noexcept {
		return this->template get<2>();
	}

private:
	template<class Original>
	struct dim_replacement;

	template<class ArgLength, class RetSig>
	struct dim_replacement<function_sig<Dim, ArgLength, RetSig>> {
		using type = function_sig<Dim, arg_length_from_t<StrideT>, RetSig>;
	};

	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static_assert(requires { StartT::value; }, "Cannot slice a tuple dimension dynamically");
		static_assert(requires { StrideT::value; }, "Cannot slice a tuple dimension dynamically");
		static constexpr std::size_t start = StartT::value;
		static constexpr std::size_t stride = StrideT::value;
		static constexpr std::size_t sub_length = sizeof...(RetSigs);

		template<class Indices = std::make_index_sequence<(sub_length + stride - start - 1) / stride>>
		struct pack_helper;

		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> {
			using type = dep_function_sig<Dim, typename original::template ret_sig<Indices * stride + start>...>;
		};

		using type = typename pack_helper<>::type;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, StartT start, StrideT stride) noexcept {
			using namespace constexpr_arithmetic;
			if constexpr (State::template contains<index_in<Dim>>) {
				return state.template with<index_in<Dim>>(state.template get<index_in<Dim>>() * stride + start);
			} else {
				return state;
			}
		}
	};

public:
	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, start(), stride());
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<length_in<Dim>, index_in<Dim>>();
	}

	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t =
		decltype(impl::sub_state(std::declval<State>(), std::declval<StartT>(), std::declval<StrideT>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set step length");
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set step length");
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, step_t, State>())
	{
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set step length");
		return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept
	requires (has_length<QDim, State>())
	{
		using namespace constexpr_arithmetic;
		if constexpr (QDim == Dim) {
			const auto sub_length = sub_structure().template length<Dim>(state);
			return (sub_length + stride() - start() - make_const<1>()) / stride();
		} else {
			return sub_structure().template length<QDim>(sub_state(state));
		}
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		static_assert(!State::template contains<length_in<Dim>>, "Cannot set step length");
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, step_t, State>())
	{
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim, class StartT, class StrideT>
struct step_proto : strict_contain<StartT, StrideT> {
	using strict_contain<StartT, StrideT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return step_t<Dim, Struct, StartT, StrideT>(s, this->template get<0>(), this->template get<1>());
	}
};

template<IsDim auto Dim, class StartT, class StrideT>
constexpr auto step(StartT start, StrideT stride) noexcept {
	return step_proto<Dim, good_index_t<StartT>, good_index_t<StrideT>>(start, stride);
}

template<class StartT, class StrideT>
struct auto_step_proto : strict_contain<StartT, StrideT> {
	using strict_contain<StartT, StrideT>::strict_contain;

	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		static_assert(
			!Struct::signature::dependent,
			"Add a dimension name as the first parameter to step, or use a structure with a dynamic topmost dimension");
		constexpr auto dim = Struct::signature::dim;
		return step_t<dim, Struct, StartT, StrideT>(s, this->template get<0>(), this->template get<1>());
	}
};

template<class StartT, class StrideT>
constexpr auto step(StartT start, StrideT stride) noexcept {
	return auto_step_proto<good_index_t<StartT>, good_index_t<StrideT>>(start, stride);
}

template<IsDim auto Dim, class T>
struct reverse_t : strict_contain<T> {
	using strict_contain<T>::strict_contain;

	static constexpr char name[] = "reverse_t";
	using params = struct_params<dim_param<Dim>, structure_param<T>>;

	template<IsState State>
	[[nodiscard]]
	constexpr T sub_structure(State /*state*/) const noexcept {
		return this->get();
	}

	[[nodiscard]]
	constexpr T sub_structure() const noexcept {
		return this->get();
	}

private:
	template<class Original>
	struct dim_replacement {
		using type = Original;
	};

	template<class... RetSigs>
	struct dim_replacement<dep_function_sig<Dim, RetSigs...>> {
		using original = dep_function_sig<Dim, RetSigs...>;
		static constexpr std::size_t len = sizeof...(RetSigs);

		template<class Indices = std::make_index_sequence<len>>
		struct pack_helper;

		template<std::size_t... Indices>
		struct pack_helper<std::index_sequence<Indices...>> {
			using type = dep_function_sig<Dim, typename original::template ret_sig<len - 1 - Indices>...>;
		};

		using type = typename pack_helper<>::type;
	};

	struct impl {
		template<IsState State>
		[[nodiscard]]
		static constexpr auto sub_state(State state, T sub_structure) noexcept {
			using namespace constexpr_arithmetic;
			if constexpr (State::template contains<index_in<Dim>>) {
				const auto tmp_state = state.template remove<index_in<Dim>>();

				if constexpr (sub_structure_t::template has_length<Dim, decltype(tmp_state)>()) {
					return tmp_state.template with<index_in<Dim>>(sub_structure.template length<Dim>(tmp_state) -
																make_const<1>() - state.template get<index_in<Dim>>());
				} else {
					return tmp_state;
				}
			} else {
				return state;
			}
		}
	};

public:
	template<IsState State>
	[[nodiscard]]
	constexpr auto sub_state(State state) const noexcept {
		return impl::sub_state(state, sub_structure());
	}

	template<IsState State>
	[[nodiscard]]
	static constexpr auto clean_state(State state) noexcept {
		return state.template remove<length_in<Dim>, index_in<Dim>>();
	}

	using signature = typename T::signature::template replace<dim_replacement, Dim>;

	using sub_structure_t = T;
	template<IsState State>
	using sub_state_t = decltype(impl::sub_state(std::declval<State>(), std::declval<T>()));
	template<IsState State>
	using clean_state_t = decltype(clean_state(std::declval<State>()));

	template<IsState State>
	[[nodiscard]]
	static constexpr bool has_size() noexcept {
		return sub_structure_t::template has_size<sub_state_t<State>>();
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto size(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().size(sub_state(state));
	}

	template<IsState State>
	[[nodiscard]]
	constexpr auto align(State state) const noexcept
	requires (has_size<State>())
	{
		return sub_structure().align(sub_state(state));
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_offset_of() noexcept {
		return has_offset_of<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_offset_of(State state) const noexcept
	requires (has_offset_of<Sub, reverse_t, State>())
	{
		return offset_of<Sub>(sub_structure(), sub_state(state));
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	static constexpr bool has_length() noexcept {
		return sub_structure_t::template has_length<QDim, sub_state_t<State>>();
	}

	template<auto QDim, IsState State>
	requires IsDim<decltype(QDim)>
	[[nodiscard]]
	constexpr auto length(State state) const noexcept {
		return sub_structure().template length<QDim>(state);
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	static constexpr bool has_strict_state_at() noexcept {
		return has_state_at<Sub, sub_structure_t, sub_state_t<State>>();
	}

	template<class Sub, IsState State>
	[[nodiscard]]
	constexpr auto strict_state_at(State state) const noexcept
	requires (has_state_at<Sub, reverse_t, State>())
	{
		return state_at<Sub>(sub_structure(), sub_state(state));
	}
};

template<IsDim auto Dim>
struct reverse_proto {
	static constexpr bool proto_preserves_layout = true;

	template<class Struct>
	[[nodiscard]]
	constexpr auto instantiate_and_construct(Struct s) const noexcept {
		return reverse_t<Dim, Struct>(s);
	}
};

/**
 * @brief reverses an index (or indices) given by dimension name(s) in a structure
 *
 * @tparam Dim: the dimension names
 */
template<auto... Dims>
requires IsDimPack<decltype(Dims)...>
constexpr auto reverse() noexcept {
	return (... ^ reverse_proto<Dims>());
}

template<>
constexpr auto reverse<>() noexcept {
	return neutral_proto();
}

} // namespace noarr

#endif // NOARR_STRUCTURES_SLICE_HPP
