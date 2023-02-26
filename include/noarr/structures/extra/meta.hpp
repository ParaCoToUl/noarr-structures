#ifndef NOARR_STRUCTURES_META_HPP
#define NOARR_STRUCTURES_META_HPP

#include "../base/utility.hpp"
#include "../structs/bcast.hpp"
#include "../structs/scalar.hpp"
#include "../structs/layouts.hpp"
#include "../structs/blocks.hpp"
#include "../structs/views.hpp"
#include "../structs/setters.hpp"

namespace noarr {

namespace helpers {

template<class>
struct state_make_bcast {
	template<class ValueType>
	static constexpr auto construct(ValueType) noexcept {
		return neutral_proto();
	}
};

template<char Dim, class ValueType>
struct state_make_bcast<state_item<length_in<Dim>, ValueType>> {
	static constexpr auto construct(ValueType) noexcept {
		return bcast<Dim>();
	}
};

template<class>
struct state_make_fix {
	template<class Bcasts, class ValueType>
	static constexpr auto construct(Bcasts, ValueType) noexcept {
		return neutral_proto();
	}
};

template<char Dim, class ValueType>
struct state_make_fix<state_item<index_in<Dim>, ValueType>> {
	template<class Bcasts>
	static constexpr auto construct(Bcasts, ValueType) noexcept {
		if constexpr (Bcasts::signature::template any_accept<Dim>)
			return neutral_proto();
		else
			return bcast<Dim>();
	}
};

}

template<class... StateItems>
constexpr auto state_consumer(state<StateItems...> state) noexcept {
	auto bcasts = (scalar<void>() ^ ... ^ helpers::state_make_bcast<StateItems>::construct(state.template get<typename StateItems::tag>()));
	auto fixes = (neutral_proto() ^ ... ^ helpers::state_make_fix<StateItems>::construct(bcasts, state.template get<typename StateItems::tag>()));
	return bcasts ^ fixes;
}

} // namespace noarr

#endif // NOARR_STRUCTURES_META_HPP
