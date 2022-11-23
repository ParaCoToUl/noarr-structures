#include <catch2/catch.hpp>

#include <array>
#include <iostream>

#include <noarr/structures_extended.hpp>
#define NOARR_DUMMY_CUDA
#include <noarr/structures/interop/cuda.hpp>

TEST_CASE("Cuda traverser simple 6D", "[cuda]") {
	auto s = noarr::scalar<int>()
		^ noarr::array<'a', 11>()
		^ noarr::array<'b', 29>()
		^ noarr::array<'c', 37>()
		^ noarr::array<'d', 53>()
		^ noarr::array<'e', 79>()
		^ noarr::array<'f', 97>()
		^ noarr::array<'g', 3>()
	;

	auto t = noarr::cuda_traverser(s).threads<'a', 'b', 'c', 'd', 'e', 'f'>();

	REQUIRE(11 == t.grid_dim().x);
	REQUIRE(29 == t.block_dim().x);
	REQUIRE(37 == t.grid_dim().y);
	REQUIRE(53 == t.block_dim().y);
	REQUIRE(79 == t.grid_dim().z);
	REQUIRE(97 == t.block_dim().z);
	
	blockIdx = {10, 30, 50};
	threadIdx = {20, 40, 60};

	std::size_t g = 0;
	t.inner().for_each([&g](auto state){
		REQUIRE(10 == state.template get<noarr::index_in<'a'>>());
		REQUIRE(20 == state.template get<noarr::index_in<'b'>>());
		REQUIRE(30 == state.template get<noarr::index_in<'c'>>());
		REQUIRE(40 == state.template get<noarr::index_in<'d'>>());
		REQUIRE(50 == state.template get<noarr::index_in<'e'>>());
		REQUIRE(60 == state.template get<noarr::index_in<'f'>>());
		REQUIRE(g == state.template get<noarr::index_in<'g'>>());
		g++;
	});
	REQUIRE(g == 3);
}

TEST_CASE("Cuda traverser simple 2D", "[cuda]") {
	auto s = noarr::scalar<int>()
		^ noarr::array<'a', 11>()
		^ noarr::array<'b', 29>()
		^ noarr::array<'g', 3>()
	;

	auto t = noarr::cuda_traverser(s).threads<'a', 'b'>();

	REQUIRE(11 == t.grid_dim().x);
	REQUIRE(29 == t.block_dim().x);
	REQUIRE(1 == t.grid_dim().y);
	REQUIRE(1 == t.block_dim().y);
	REQUIRE(1 == t.grid_dim().z);
	REQUIRE(1 == t.block_dim().z);
	
	blockIdx = {10};
	threadIdx = {20};

	std::size_t g = 0;
	t.inner().for_each([&g](auto state){
		REQUIRE(10 == state.template get<noarr::index_in<'a'>>());
		REQUIRE(20 == state.template get<noarr::index_in<'b'>>());
		REQUIRE(g == state.template get<noarr::index_in<'g'>>());
		g++;
	});
	REQUIRE(g == 3);
}

TEST_CASE("Cuda traverser confusing 2D", "[cuda]") {
	auto s = noarr::scalar<int>()
		^ noarr::array<'a', 11>()
		^ noarr::array<'b', 29>()
		^ noarr::array<'g', 3>()
	;

	auto t = noarr::cuda_traverser(s).order(noarr::rename<'a', 'c'>()).threads<'c', 'b'>();

	REQUIRE(11 == t.grid_dim().x);
	REQUIRE(29 == t.block_dim().x);
	REQUIRE(1 == t.grid_dim().y);
	REQUIRE(1 == t.block_dim().y);
	REQUIRE(1 == t.grid_dim().z);
	REQUIRE(1 == t.block_dim().z);
	
	blockIdx = {10};
	threadIdx = {20};

	std::size_t g = 0;
	t.inner().order(noarr::rename<'g', 'h'>()).for_each([&g](auto state){
		REQUIRE(10 == state.template get<noarr::index_in<'a'>>());
		REQUIRE(20 == state.template get<noarr::index_in<'b'>>());
		REQUIRE(g == state.template get<noarr::index_in<'g'>>());
		g++;
	});
	REQUIRE(g == 3);
}

TEST_CASE("Cuda blocks", "[cuda]") {
	auto s = noarr::scalar<int>()
		^ noarr::array<'x', 800>()
		^ noarr::array<'y', 600>()
		^ noarr::array<'c', 3>()
	;

	auto t = noarr::cuda_traverser(s).order(
		noarr::into_blocks<'x', 'X', 'x'>() ^ noarr::set_length<'x'>(4) ^
		noarr::into_blocks<'y', 'Y', 'y'>() ^ noarr::set_length<'y'>(8)
	).threads<'X', 'x', 'Y', 'y'>();

	REQUIRE(800/4 == t.grid_dim().x);
	REQUIRE(4 == t.block_dim().x);
	REQUIRE(600/8 == t.grid_dim().y);
	REQUIRE(8 == t.block_dim().y);
	REQUIRE(1 == t.grid_dim().z);
	REQUIRE(1 == t.block_dim().z);
	
	blockIdx = {10, 20};
	threadIdx = {3, 5};

	std::size_t c = 0;
	t.inner().for_each([&c](auto state){
		REQUIRE(10*4+3 == state.template get<noarr::index_in<'x'>>());
		REQUIRE(20*8+5 == state.template get<noarr::index_in<'y'>>());
		REQUIRE(c == state.template get<noarr::index_in<'c'>>());
		c++;
	});
	REQUIRE(c == 3);
}
