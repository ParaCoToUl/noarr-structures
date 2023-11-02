#include <noarr_test/macros.hpp>

#include <type_traits>

#include <noarr/structures_extended.hpp>

TEST_CASE("Simple rename use-case", "[rename]") {
	using namespace noarr;

	auto structure = array_t<'x', 10, array_t<'y', 20, scalar<int>>>();
	auto structure_renamed = structure ^ rename<'x', 'z'>() ^ rename<'y', 'x'>() ^ rename<'z', 'y'>();

	REQUIRE(std::is_same_v<typename decltype(structure_renamed)::signature, typename array_t<'y', 10, array_t<'x', 20, scalar<int>>>::signature>);
	REQUIRE((structure | offset<'x', 'y'>(3, 5)) == (structure_renamed | offset<'y', 'x'>(3, 5)));
}

TEST_CASE("Rename with reorder", "[rename]") {
	using namespace noarr;

	auto structure = array_t<'x', 10, array_t<'y', 20, scalar<int>>>();
	auto structure_renamed = structure ^ rename<'x', 'z'>() ^ rename<'y', 'x'>() ^ rename<'z', 'y'>() ^ reorder<'x', 'y'>();

	REQUIRE(std::is_same_v<typename decltype(structure_renamed)::signature, typename array_t<'x', 20, array_t<'y', 10, scalar<int>>>::signature>);
	REQUIRE((structure | offset<'x', 'y'>(3, 5)) == (structure_renamed | offset<'y', 'x'>(3, 5)));
}

TEST_CASE("Atomic rename", "[rename]") {
	using namespace noarr;

	auto structure = array_t<'x', 10, array_t<'y', 20, scalar<int>>>();
	auto structure_renamed = structure ^ rename<'x', 'y', 'y', 'x'>();

	REQUIRE(std::is_same_v<typename decltype(structure_renamed)::signature, typename array_t<'y', 10, array_t<'x', 20, scalar<int>>>::signature>);
	REQUIRE((structure | offset<'x', 'y'>(3, 5)) == (structure_renamed | offset<'y', 'x'>(3, 5)));
}

TEST_CASE("Rename in tuple", "[rename]") {
	using namespace noarr;

	auto structure = tuple_t<'t', array_t<'x', 10, array_t<'y', 20, scalar<int>>>, array_t<'x', 10, array_t<'y', 20, scalar<int>>>>();

	auto structure_renamed = structure ^ rename<'x', 'z'>() ^ rename<'y', 'x'>() ^ rename<'z', 'y'>();
	auto structure_renamed2 = structure ^ rename<'x', 'y', 't', 's', 'y', 'x'>();

	REQUIRE(std::is_same_v<typename decltype(structure_renamed)::signature, typename tuple_t<'t', array_t<'y', 10, array_t<'x', 20, scalar<int>>>, array_t<'y', 10, array_t<'x', 20, scalar<int>>>>::signature>);
	REQUIRE(std::is_same_v<typename decltype(structure_renamed2)::signature, typename tuple_t<'s', array_t<'y', 10, array_t<'x', 20, scalar<int>>>, array_t<'y', 10, array_t<'x', 20, scalar<int>>>>::signature>);

	REQUIRE((structure | offset<'t', 'x', 'y'>(lit<0>, 3, 5)) == (structure_renamed | offset<'t', 'y', 'x'>(lit<0>, 3, 5)));
	REQUIRE((structure | offset<'t', 'x', 'y'>(lit<1>, 3, 5)) == (structure_renamed | offset<'t', 'y', 'x'>(lit<1>, 3, 5)));

	REQUIRE((structure | offset<'t', 'x', 'y'>(lit<0>, 3, 5)) == (structure_renamed2 | offset<'s', 'y', 'x'>(lit<0>, 3, 5)));
	REQUIRE((structure | offset<'t', 'x', 'y'>(lit<1>, 3, 5)) == (structure_renamed2 | offset<'s', 'y', 'x'>(lit<1>, 3, 5)));
}

TEST_CASE("Rename ignores overriden dimensions", "[rename]") {
	using namespace noarr;

	auto structure = array_t<'x', 10, array_t<'y', 20, scalar<int>>>();
	auto structure_renamed = structure ^ rename<'x', 'z'>() ^ rename<'y', 'w'>();

	REQUIRE(std::is_same_v<typename decltype(structure_renamed)::signature, typename array_t<'z', 10, array_t<'w', 20, scalar<int>>>::signature>);

	auto gold_offset = (structure_renamed | offset<'w', 'z'>(3, 5));
	REQUIRE((structure | offset<'x', 'y'>(5, 3)) == gold_offset);

	REQUIRE(gold_offset == (structure_renamed | offset<'w', 'z', 'x', 'y'>(3, 5, 0, 0)));
	REQUIRE(gold_offset == (structure_renamed | offset<'x', 'y', 'w', 'z'>(0, 0, 3, 5)));
	REQUIRE(gold_offset == (structure_renamed | offset<'w', 'x', 'y', 'z'>(3, 0, 0, 5)));

	REQUIRE(gold_offset == (structure_renamed ^ fix<'x', 'y'>(0, 0) | offset<'w', 'z'>(3, 5)));
	REQUIRE(gold_offset == (structure_renamed ^ fix<'w', 'z'>(3, 5) | offset<'x', 'y'>(0, 0)));

	REQUIRE(gold_offset == (structure_renamed ^ fix<'x', 'y'>(0, 0) | offset(idx<'w', 'z'>(3, 5))));
	REQUIRE(gold_offset == (structure_renamed ^ fix<'w', 'z'>(3, 5) | offset(idx<'x', 'y'>(0, 0))));

	REQUIRE(gold_offset == (structure_renamed | offset(idx<'w', 'z', 'x', 'y'>(3, 5, 0, 0))));
	REQUIRE(gold_offset == (structure_renamed | offset(idx<'x', 'y', 'w', 'z'>(0, 0, 3, 5))));
	REQUIRE(gold_offset == (structure_renamed | offset(idx<'w', 'x', 'y', 'z'>(3, 0, 0, 5))));
}
