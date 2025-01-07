#ifndef NOARR_STRUCTURES_INTEROP_STD_PARA_HPP
#define NOARR_STRUCTURES_INTEROP_STD_PARA_HPP

#include <algorithm>
#include <execution>

#include "../interop/planner_iter.hpp"
#include "../interop/traverser_iter.hpp"

namespace noarr {

template<IsTraverser Traverser, class F>
inline void std_for_each(const Traverser &t, const F &f) {
	std::for_each(std::execution::par, t.begin(), t.end(), [&f](const auto &t_inner) { t_inner.for_each(f); });
}

template<IsTraverser Traverser, class F>
inline void std_for_sections(const Traverser &t, const F &f) {
	std::for_each(std::execution::par, t.begin(), t.end(), [&f](const auto &t_inner) { f(t_inner); });
}

struct planner_std_execute_t {};

constexpr planner_std_execute_t planner_std_execute() noexcept { return {}; }

template<IsPlanner Planner>
inline void operator|(const Planner &planner, planner_std_execute_t /*unused*/) {
	std::for_each(std::execution::par, planner.begin(), planner.end(),
	              [](const auto &inner_planner) { inner_planner.execute(); });
}

} // namespace noarr

#endif // NOARR_STRUCTURES_INTEROP_STD_PARA_HPP
