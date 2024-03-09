#include <noarr_test/macros.hpp>

#include <type_traits>

#include <noarr/structures_extended.hpp>
#include <noarr/structures/extra/traverser.hpp>
#include <noarr/structures/interop/bag.hpp>

namespace {
	constexpr noarr::dim<__LINE__> x;
	constexpr noarr::dim<__LINE__> X;
	constexpr noarr::dim<__LINE__> x_guard;

	constexpr noarr::dim<__LINE__> y;
	constexpr noarr::dim<__LINE__> Y;
	constexpr noarr::dim<__LINE__> y_guard;
}

TEST_CASE("Simple use-case of dims", "[dim]") {
	using namespace noarr;

	auto structure = array_t<x, 10, array_t<y, 20, scalar<int>>>();

	// can appear in signatures
	REQUIRE(std::is_same_v<typename decltype(structure)::signature, typename array_t<x, 10, array_t<y, 20, scalar<int>>>::signature>);

	// can appear in offsets
	REQUIRE((structure | offset<x, y>(3, 5)) == (structure | offset<x, y>(3, 5)));

	std::size_t iterations = 0;

	traverser(structure).template for_each<x, y>([=, &iterations](auto state) {
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure).template for_each<y, x>([=, &iterations](auto state) {
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_each<x, y>([=, &iterations](auto state) {
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_each<y, x>([=, &iterations](auto state) {
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure).template for_dims<x, y>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure).template for_dims<y, x>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == j * 10 + i);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_dims<x, y>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_dims<y, x>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == j * 10 + i);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure).template for_sections<x, y>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure).template for_sections<y, x>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_sections<x, y>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});

	REQUIRE(iterations == 10 * 20);
	iterations = 0;

	traverser(structure) | for_sections<y, x>([=, &iterations](auto inner) {
		auto state = inner.state();
		auto [i, j] = noarr::get_indices<x, y>(state);

		REQUIRE(iterations++ == i * 20 + j);
	});
}

TEST_CASE("into_blocks_dynamic use-case of dims", "[dim]") {
	using namespace noarr;

	auto structure = scalar<int>() ^ array<x, 10>();
	auto structure_blocked = structure ^ into_blocks_dynamic<x, X, x, x_guard>(4);

	std::size_t outer_iterations = 0, inner_iterations = 0;

	traverser(structure_blocked)
		.template for_dims<X, x>([=, &outer_iterations, &inner_iterations](auto inner) {
			auto state = inner.state();
			auto [X_idx, x_idx] = get_indices<X, x>(state);

			inner.template for_each<x_guard>([=, &inner_iterations](auto state) {
				REQUIRE(get_index<x_guard>(state) == 0);

				REQUIRE((structure_blocked | offset(state)) < 10 * sizeof(int));

				++inner_iterations;
			});

			REQUIRE(X_idx < 3);
			REQUIRE(x_idx < 4);

			++outer_iterations;
		});

	REQUIRE(outer_iterations == 12);
	REQUIRE(inner_iterations == 10);

	traverser(structure)
		.order(into_blocks_dynamic<x, X, x, x_guard>(4))
		.template for_dims<X, x>([=, &outer_iterations, &inner_iterations](auto inner) {
			inner.template for_each<x_guard>([=, &inner_iterations](auto state) {
				REQUIRE(get_index<x>(state) < 10);

				REQUIRE((structure | offset(state)) < 10 * sizeof(int));

				--inner_iterations;
			});

			--outer_iterations;
		});

	REQUIRE(outer_iterations == 0);
	REQUIRE(inner_iterations == 0);
}

TEST_CASE("into_blocks_dynamic tiling use-case of dims", "[dim]") {
	using namespace noarr;

	auto structure = scalar<int>() ^ array<x, 10>() ^ array<y, 10>();
	auto structure_blocked = structure ^ into_blocks_dynamic<x, X, x, x_guard>(4) ^ into_blocks_dynamic<y, Y, y, y_guard>(4);

	std::size_t outer_iterations = 0, inner_iterations = 0;

	traverser(structure_blocked)
		.template for_dims<X, x, Y, y>([=, &outer_iterations, &inner_iterations](auto inner) {
			auto state = inner.state();
			auto [X_idx, x_idx] = get_indices<X, x>(state);
			auto [Y_idx, y_idx] = get_indices<Y, y>(state);

			inner.template for_each<x_guard, y_guard>([=, &inner_iterations](auto state) {
				REQUIRE(get_index<x_guard>(state) == 0);
				REQUIRE(get_index<y_guard>(state) == 0);

				REQUIRE((structure_blocked | offset(state)) < 100 * sizeof(int));

				++inner_iterations;
			});

			REQUIRE(X_idx < 3);
			REQUIRE(x_idx < 4);

			REQUIRE(Y_idx < 3);
			REQUIRE(y_idx < 4);

			++outer_iterations;
		});

	REQUIRE(outer_iterations == 144);
	REQUIRE(inner_iterations == 100);

	traverser(structure)
		.order(into_blocks_dynamic<x, X, x, x_guard>(4))
		.order(into_blocks_dynamic<y, Y, y, y_guard>(4))
		.template for_dims<X, x, Y, y>([=, &outer_iterations, &inner_iterations](auto inner) {
			inner.template for_each<x_guard, y_guard>([=, &inner_iterations](auto state) {
				REQUIRE(get_index<x>(state) < 10);
				REQUIRE(get_index<y>(state) < 10);

				REQUIRE((structure | offset(state)) < 100 * sizeof(int));

				--inner_iterations;
			});

			--outer_iterations;
		});

	REQUIRE(outer_iterations == 0);
	REQUIRE(inner_iterations == 0);
}

namespace {

void compute_kernel(auto A, auto B, auto E) {
	using namespace noarr;

	{
		traverser(A, B, E)
			.order(into_blocks_dynamic<'x', 'X', 'x', x_guard>(4))
			.order(into_blocks_dynamic<'y', 'Y', 'y', y_guard>(4))
			.template for_dims<'x', 'y', 'X', 'Y'>([=](auto inner) {
				inner.template for_each<x_guard, y_guard>([=](auto state) {
					REQUIRE(get_index<'x'>(state) < 10);
					REQUIRE(get_index<'y'>(state) < 10);

					REQUIRE((E | offset(state)) < 100 * sizeof(int));
				});
			});
	}
}

}

TEST_CASE("into_blocks_dynamic bug use-case of dims", "[dim]") {
	using namespace noarr;

	auto E = noarr::make_bag(scalar<int>() ^ sized_vectors<'x', 'y'>(10, 10));
	auto A = noarr::make_bag(scalar<int>() ^ sized_vectors<'x', 'z'>(10, 10));
	auto B = noarr::make_bag(scalar<int>() ^ sized_vectors<'z', 'y'>(10, 10));

	compute_kernel(A.get_ref(), B.get_ref(), E.get_ref());

}
