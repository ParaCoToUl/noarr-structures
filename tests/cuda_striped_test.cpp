#include <noarr_test/macros.hpp>

#include <cstdint>

#include <noarr/structures_extended.hpp>
#include "noarr_test_cuda_dummy.hpp"
#include <noarr/structures/interop/cuda_striped.cuh>

TEST_CASE("Cuda striped - 32bit scalar array", "[cuda]") {
	static_assert(sizeof(std::uint32_t) == 4);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 32;
	constexpr std::size_t stripe_size = 4;

	auto s = noarr::scalar<std::uint32_t>() ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes>();

	STATIC_REQUIRE((s | noarr::get_size()) == 1000 * nstripes * sizeof(std::uint32_t));
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	for(std::size_t j = 0; j < nstripes; j++) {
		for(std::size_t i = 0; i < 100; i++) {
			REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(i, j))) == i*period + j*stripe_size);
		}
	}
}

TEST_CASE("Cuda striped - 64bit scalar array", "[cuda]") {
	static_assert(sizeof(std::uint64_t) == 8);

	constexpr std::size_t throughput = 2;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 32;
	constexpr std::size_t stripe_size = 8;

	auto s = noarr::scalar<std::uint64_t>() ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes>();

	REQUIRE((s | noarr::get_size()) == 1000 * nstripes * sizeof(std::uint64_t));
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	for(std::size_t j = 0; j < nstripes; j++) {
		for(std::size_t i = 0; i < 100; i++) {
			REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(i, j))) == i*period + j*stripe_size);
		}
	}
}

TEST_CASE("Cuda striped - 16bit scalar array", "[cuda]") {
	static_assert(sizeof(std::uint16_t) == 2);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 32;
	constexpr std::size_t stripe_size = 4;

	auto s = noarr::scalar<std::uint16_t>() ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes>();

	STATIC_REQUIRE((s | noarr::get_size()) == 1000 * nstripes * sizeof(std::uint16_t));
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 0))) == 0*period + 0*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 0))) == 0*period + 0*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 0))) == 1*period + 0*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 0))) == 1*period + 0*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 0))) == 2*period + 0*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 0))) == 2*period + 0*stripe_size + 1*sizeof(std::uint16_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(0))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == 0*period + 1*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == 0*period + 1*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 1))) == 1*period + 1*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 1))) == 1*period + 1*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 1))) == 2*period + 1*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == 2*period + 1*stripe_size + 1*sizeof(std::uint16_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(1))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == 0*period + 2*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == 0*period + 2*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 2))) == 1*period + 2*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 2))) == 1*period + 2*stripe_size + 1*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 2))) == 2*period + 2*stripe_size + 0*sizeof(std::uint16_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == 2*period + 2*stripe_size + 1*sizeof(std::uint16_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(2))));
}

TEST_CASE("Cuda striped - 8bit scalar array", "[cuda]") {
	static_assert(sizeof(std::uint8_t) == 1);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 32;
	constexpr std::size_t stripe_size = 4;

	auto s = noarr::scalar<std::uint8_t>() ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes>();

	STATIC_REQUIRE((s | noarr::get_size()) == 1000 * nstripes * sizeof(std::uint8_t));
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 0))) == 0*period + 0*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 0))) == 0*period + 0*stripe_size + 1*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 0))) == 0*period + 0*stripe_size + 2*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 0))) == 0*period + 0*stripe_size + 3*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 0))) == 1*period + 0*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 0))) == 1*period + 0*stripe_size + 1*sizeof(std::uint8_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(0))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == 0*period + 1*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == 0*period + 1*stripe_size + 1*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 1))) == 0*period + 1*stripe_size + 2*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 1))) == 0*period + 1*stripe_size + 3*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 1))) == 1*period + 1*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == 1*period + 1*stripe_size + 1*sizeof(std::uint8_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(1))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == 0*period + 2*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == 0*period + 2*stripe_size + 1*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 2))) == 0*period + 2*stripe_size + 2*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 2))) == 0*period + 2*stripe_size + 3*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 2))) == 1*period + 2*stripe_size + 0*sizeof(std::uint8_t));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == 1*period + 2*stripe_size + 1*sizeof(std::uint8_t));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(2, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(2) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(3, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(3) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(4, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(4) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(2))));
}

TEST_CASE("Cuda striped - 24bit scalar array - 6 stripes", "[cuda]") {
	struct color { std::uint8_t r, g, b; };

	static_assert(sizeof(color) == 3);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 6;
	constexpr std::size_t stripe_size = sizeof(color)*6 + 2; // 6 elems per stripe, 2 bytes padding (to get 4-byte alignment)

	auto s = noarr::scalar<color>() ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes>();

	STATIC_REQUIRE((s | noarr::get_size()) == 167 * period); // 167 = ceil(1000 / 6), where 6 = number of elems per stripe and period
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 0, 0))) == 0*period + 0*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 1, 0))) == 0*period + 0*stripe_size + 1*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 2, 0))) == 0*period + 0*stripe_size + 2*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 3, 0))) == 0*period + 0*stripe_size + 3*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 4, 0))) == 0*period + 0*stripe_size + 4*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 5, 0))) == 0*period + 0*stripe_size + 5*sizeof(color));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 0, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 0) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 1, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 1) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 2, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 2) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 3, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 3) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 4, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 4) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 5, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 5) + noarr::cuda_stripe_idx(0))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 6, 0))) == 1*period + 0*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(11, 0))) == 1*period + 0*stripe_size + 5*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(12, 0))) == 2*period + 0*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(17, 0))) == 2*period + 0*stripe_size + 5*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(18, 0))) == 3*period + 0*stripe_size + 0*sizeof(color));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>( 6, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 6) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(11, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(11) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(12, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(12) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(17, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(17) + noarr::cuda_stripe_idx(0))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(18, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(18) + noarr::cuda_stripe_idx(0))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == 0*period + 1*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == 0*period + 1*stripe_size + 1*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == 0*period + 1*stripe_size + 5*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 1))) == 1*period + 1*stripe_size + 0*sizeof(color));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(1))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::cuda_stripe_idx(1))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == 0*period + 2*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == 0*period + 2*stripe_size + 1*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == 0*period + 2*stripe_size + 5*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 2))) == 1*period + 2*stripe_size + 0*sizeof(color));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(2))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::cuda_stripe_idx(2))));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 11))) == 0*period + 11*stripe_size + 0*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 11))) == 0*period + 11*stripe_size + 1*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 11))) == 0*period + 11*stripe_size + 5*sizeof(color));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 11))) == 1*period + 11*stripe_size + 0*sizeof(color));

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(0, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::cuda_stripe_idx(11))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(1, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::cuda_stripe_idx(11))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(5, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::cuda_stripe_idx(11))));
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::cuda_stripe_index>(6, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::cuda_stripe_idx(11))));
}

TEST_CASE("Cuda striped - 24bit non-scalar array - 6 stripes", "[cuda]") {
	static_assert(sizeof(std::uint8_t) == 1);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 6;
	constexpr std::size_t stripe_size = 3*6 + 2; // 6 elems per stripe, 2 bytes padding (to get 4-byte alignment)

	auto color_s = noarr::scalar<std::uint8_t>() ^ noarr::array<'c', 3>();
	auto s = color_s ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes, decltype(color_s)>();

	STATIC_REQUIRE((s | noarr::get_size()) == 167 * period); // 167 = ceil(1000 / 6), where 6 = number of elems per stripe and period
	STATIC_REQUIRE(s.max_conflict_size == throughput);

	for(int c = 0; c < 3; c++) {
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 0, c, 0))) == 0*period + 0*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 1, c, 0))) == 0*period + 0*stripe_size + 1*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 2, c, 0))) == 0*period + 0*stripe_size + 2*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 3, c, 0))) == 0*period + 0*stripe_size + 3*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 4, c, 0))) == 0*period + 0*stripe_size + 4*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 5, c, 0))) == 0*period + 0*stripe_size + 5*3 + c);

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 0, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 0) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 1, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 1) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 2, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 2) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 3, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 3) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 6, c, 0))) == 1*period + 0*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(11, c, 0))) == 1*period + 0*stripe_size + 5*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(12, c, 0))) == 2*period + 0*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(17, c, 0))) == 2*period + 0*stripe_size + 5*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(18, c, 0))) == 3*period + 0*stripe_size + 0*3 + c);

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>( 6, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>( 6) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(11, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(11) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(12, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(12) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(17, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(17) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(18, c, 0))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(18) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(0))));

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 1))) == 0*period + 1*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 1))) == 0*period + 1*stripe_size + 1*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 1))) == 0*period + 1*stripe_size + 5*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 1))) == 1*period + 1*stripe_size + 0*3 + c);

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(1))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(1))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(1))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 1))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(1))));

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 2))) == 0*period + 2*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 2))) == 0*period + 2*stripe_size + 1*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 2))) == 0*period + 2*stripe_size + 5*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 2))) == 1*period + 2*stripe_size + 0*3 + c);

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(2))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(2))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(2))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 2))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(2))));

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 11))) == 0*period + 11*stripe_size + 0*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 11))) == 0*period + 11*stripe_size + 1*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 11))) == 0*period + 11*stripe_size + 5*3 + c);
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 11))) == 1*period + 11*stripe_size + 0*3 + c);

		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(0, c, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(0) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(11))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(1, c, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(1) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(11))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(5, c, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(5) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(11))));
		REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(6, c, 11))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(6) + noarr::idx<'c'>(c) + noarr::cuda_stripe_idx(11))));
	}

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(11, 1, 7))) == 1*period + 7*stripe_size + 5*3 + 1);
	STATIC_REQUIRE((s | noarr::offset<decltype(color_s)>(noarr::empty_state.with<noarr::index_in<'x'>,    noarr::cuda_stripe_index>(11,    7))) == 1*period + 7*stripe_size + 5*3);

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(11, 1, 7))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(11) + noarr::idx<'c'>(1) + noarr::cuda_stripe_idx(7))));
	STATIC_REQUIRE((s | noarr::offset<decltype(color_s)>(noarr::empty_state.with<noarr::index_in<'x'>,    noarr::cuda_stripe_index>(11,    7))) == (s | noarr::offset<decltype(color_s)>(noarr::empty_state + noarr::idx<'x'>(11) + noarr::cuda_stripe_idx(7))));
}

TEST_CASE("Cuda striped constexpr arithmetic", "[cuda cearithm]") {
	static_assert(sizeof(std::uint8_t) == 1);

	constexpr std::size_t throughput = 1;
	constexpr std::size_t period = 4*32*throughput;
	constexpr std::size_t nstripes = 6;
	constexpr std::size_t stripe_size = 3*6 + 2; // 6 elems per stripe, 2 bytes padding (to get 4-byte alignment)

	auto color_s = noarr::scalar<std::uint8_t>() ^ noarr::array<'c', 3>();
	auto s = color_s ^ noarr::array<'x', 1000>() ^ noarr::cuda_striped<nstripes, decltype(color_s)>();

	STATIC_REQUIRE((s | noarr::get_size()).value == 167 * period); // 167 = ceil(1000 / 6), where 6 = number of elems per stripe and period
	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(noarr::lit<11>, noarr::lit<1>, noarr::lit<7>))).value == 1*period + 7*stripe_size + 5*3 + 1);
	STATIC_REQUIRE((s | noarr::offset<decltype(color_s)>(noarr::empty_state.with<noarr::index_in<'x'>,    noarr::cuda_stripe_index>(noarr::lit<11>,                noarr::lit<7>))).value == 1*period + 7*stripe_size + 5*3);

	STATIC_REQUIRE((s | noarr::offset(noarr::empty_state.with<noarr::index_in<'x'>, noarr::index_in<'c'>, noarr::cuda_stripe_index>(noarr::lit<11>, noarr::lit<1>, noarr::lit<7>))) == (s | noarr::offset(noarr::empty_state + noarr::idx<'x'>(noarr::lit<11>) + noarr::idx<'c'>(noarr::lit<1>) + noarr::cuda_stripe_idx(noarr::lit<7>))));
	STATIC_REQUIRE((s | noarr::offset<decltype(color_s)>(noarr::empty_state.with<noarr::index_in<'x'>,    noarr::cuda_stripe_index>(noarr::lit<11>,                noarr::lit<7>))) == (s | noarr::offset<decltype(color_s)>(noarr::empty_state + noarr::idx<'x'>(noarr::lit<11>) + noarr::cuda_stripe_idx(noarr::lit<7>))));
}
