#include <noarr_test/macros.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/shortcuts.hpp>
#include "noarr_test_defs.hpp"

TEST_CASE("State ordering - one dimension", "[state]") {
	STATIC_REQUIRE(noarr::idx<'x'>(10) < noarr::idx<'x'>(20));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'x'>(20));

	STATIC_REQUIRE(!(noarr::idx<'x'>(20) < noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(20) <= noarr::idx<'x'>(10)));

	STATIC_REQUIRE(noarr::idx<'x'>(20) > noarr::idx<'x'>(10));
	STATIC_REQUIRE(noarr::idx<'x'>(20) >= noarr::idx<'x'>(10));

	STATIC_REQUIRE(!(noarr::idx<'x'>(10) > noarr::idx<'x'>(20)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) >= noarr::idx<'x'>(20)));

	STATIC_REQUIRE(noarr::idx<'x'>(10) == noarr::idx<'x'>(10));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'x'>(10));
	STATIC_REQUIRE(noarr::idx<'x'>(10) >= noarr::idx<'x'>(10));

	STATIC_REQUIRE(!(noarr::idx<'x'>(10) != noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) < noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) > noarr::idx<'x'>(10)));

	STATIC_REQUIRE(noarr::idx<'x'>(10) != noarr::idx<'x'>(20));
}

TEST_CASE("State ordering - different dimensions", "[state]") {
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) < noarr::idx<'y'>(20)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) <= noarr::idx<'y'>(20)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) > noarr::idx<'y'>(20)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) >= noarr::idx<'y'>(20)));

	STATIC_REQUIRE(!(noarr::idx<'y'>(20) < noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'y'>(20) <= noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'y'>(20) > noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'y'>(20) >= noarr::idx<'x'>(10)));

	STATIC_REQUIRE(!(noarr::idx<'x'>(10) == noarr::idx<'y'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) < noarr::idx<'y'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) > noarr::idx<'y'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) <= noarr::idx<'y'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) >= noarr::idx<'y'>(10)));

	STATIC_REQUIRE(noarr::idx<'x'>(10) != noarr::idx<'y'>(20));
}

TEST_CASE("State ordering - one and two dimensions", "[state]") {
	STATIC_REQUIRE(noarr::idx<'x'>(10) < noarr::idx<'x', 'y'>(20, 20));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'x', 'y'>(20, 20));

	STATIC_REQUIRE(noarr::idx<'x'>(10) < noarr::idx<'y', 'x'>(20, 20));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'y', 'x'>(20, 20));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'x'>(20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'x'>(20)));

	STATIC_REQUIRE(!(noarr::idx<'y', 'x'>(20, 10) < noarr::idx<'x'>(10)));
	STATIC_REQUIRE(!(noarr::idx<'y', 'x'>(20, 10) <= noarr::idx<'x'>(10)));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) > noarr::idx<'x'>(10));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) >= noarr::idx<'x'>(10));

	STATIC_REQUIRE(noarr::idx<'y', 'x'>(20, 20) > noarr::idx<'x'>(10));
	STATIC_REQUIRE(noarr::idx<'y', 'x'>(20, 20) >= noarr::idx<'x'>(10));

	STATIC_REQUIRE(!(noarr::idx<'x'>(20) > noarr::idx<'x', 'y'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(20) >= noarr::idx<'x', 'y'>(10, 20)));

	STATIC_REQUIRE(!(noarr::idx<'x'>(20) > noarr::idx<'y', 'x'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x'>(20) >= noarr::idx<'y', 'x'>(20, 10)));

	STATIC_REQUIRE(!(noarr::idx<'x'>(10) == noarr::idx<'x', 'y'>(10, 20)));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'x', 'y'>(10, 20));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) >= noarr::idx<'x', 'y'>(10, 20)));

	STATIC_REQUIRE(!(noarr::idx<'x'>(10) == noarr::idx<'y', 'x'>(20, 10)));
	STATIC_REQUIRE(noarr::idx<'x'>(10) <= noarr::idx<'y', 'x'>(20, 10));
	STATIC_REQUIRE(!(noarr::idx<'x'>(10) >= noarr::idx<'y', 'x'>(20, 10)));

	STATIC_REQUIRE(noarr::idx<'x'>(10) != noarr::idx<'x', 'y'>(10, 20));
	STATIC_REQUIRE(noarr::idx<'x'>(10) != noarr::idx<'y', 'x'>(20, 10));
}

TEST_CASE("State ordering - two dimensions", "[state]") {
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'x', 'y'>(20, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'x', 'y'>(20, 20));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'y', 'x'>(20, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'y', 'x'>(20, 20));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(20, 20) < noarr::idx<'x', 'y'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(20, 20) <= noarr::idx<'x', 'y'>(10, 20)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(20, 20) < noarr::idx<'y', 'x'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(20, 20) <= noarr::idx<'y', 'x'>(20, 10)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'x', 'y'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'x', 'y'>(20, 10)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'y', 'x'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'y', 'x'>(10, 20)));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) > noarr::idx<'x', 'y'>(10, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) >= noarr::idx<'x', 'y'>(10, 20));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) > noarr::idx<'y', 'x'>(20, 10));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(20, 20) >= noarr::idx<'y', 'x'>(20, 10));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'x', 'y'>(20, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'x', 'y'>(20, 20)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'y', 'x'>(20, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'y', 'x'>(20, 20)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'x', 'y'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'x', 'y'>(20, 10)));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'y', 'x'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'y', 'x'>(10, 20)));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) == noarr::idx<'x', 'y'>(10, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'x', 'y'>(10, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'x', 'y'>(10, 20));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) != noarr::idx<'x', 'y'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'x', 'y'>(10, 20)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'x', 'y'>(10, 20)));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) == noarr::idx<'y', 'x'>(20, 10));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) <= noarr::idx<'y', 'x'>(20, 10));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) >= noarr::idx<'y', 'x'>(20, 10));

	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) != noarr::idx<'y', 'x'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) < noarr::idx<'y', 'x'>(20, 10)));
	STATIC_REQUIRE(!(noarr::idx<'x', 'y'>(10, 20) > noarr::idx<'y', 'x'>(20, 10)));

	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) != noarr::idx<'x', 'y'>(20, 20));
	STATIC_REQUIRE(noarr::idx<'x', 'y'>(10, 20) != noarr::idx<'y', 'x'>(20, 20));
}

TEST_CASE("State arithmetic - trivial", "[state]") {
	constexpr auto s1 = noarr::make_state<noarr::index_in<'x'>, noarr::index_in<'y'>>(10, 20);
	constexpr auto s2 = noarr::idx<'x'>(10) + noarr::idx<'y'>(20);
	constexpr auto s3 = noarr::idx<'x', 'y'>(0, 0);
	constexpr auto s4 = noarr::empty_state;

	STATIC_REQUIRE(std::is_same_v<decltype(s1), decltype(s2)>);

	STATIC_REQUIRE(s1 == s2);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s2)>, std::remove_cvref_t<decltype(s2 + s3)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s1)>, std::remove_cvref_t<decltype(s1 + s4)>>);

	STATIC_REQUIRE(s2 == s2 + s3);
	STATIC_REQUIRE(s1 == s1 + s4);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s3)>, std::remove_cvref_t<decltype(s1 - s1)>>);

	STATIC_REQUIRE(s3 == s1 - s1);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s4)>, std::remove_cvref_t<decltype(s4 + s4)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s4)>, std::remove_cvref_t<decltype(s4 - s4)>>);

	STATIC_REQUIRE(s4 == s4 + s4);
	STATIC_REQUIRE(s4 == s4 - s4);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s4)>, std::remove_cvref_t<decltype(+s4)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s4)>, std::remove_cvref_t<decltype(-s4)>>);

	STATIC_REQUIRE(s4 == +s4);
	STATIC_REQUIRE(s4 == -s4);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s3)>, std::remove_cvref_t<decltype(s3 + s3)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s3)>, std::remove_cvref_t<decltype(s3 - s3)>>);

	STATIC_REQUIRE(s3 == s3 + s3);
	STATIC_REQUIRE(s3 == s3 - s3);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s3)>, std::remove_cvref_t<decltype(+s3)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(s3)>, std::remove_cvref_t<decltype(-s3)>>);

	STATIC_REQUIRE(s3 == +s3);
	STATIC_REQUIRE(s3 == -s3);

	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(+s2)>, std::remove_cvref_t<decltype(s2)>>);
	STATIC_REQUIRE(std::is_same_v<std::remove_cvref_t<decltype(-s2)>, std::remove_cvref_t<decltype(s3 - s2)>>);

	STATIC_REQUIRE(+s2 == s2);
	STATIC_REQUIRE(-s2 == s3 - s2);
}

TEST_CASE("State arithmetic - non-trivial", "[state]") {
	constexpr auto s1 = noarr::idx<'x'>(10) + noarr::idx<'y'>(20);
	constexpr auto s2 = noarr::idx<'x'>(20) + noarr::idx<'z'>(30);
	constexpr auto s3 = noarr::idx<'y'>(20) + noarr::idx<'z'>(30);

	STATIC_REQUIRE(s1 + s2 == noarr::idx<'x'>(30) + noarr::idx<'y'>(20) + noarr::idx<'z'>(30));
	STATIC_REQUIRE(s1 + s3 == noarr::idx<'x'>(10) + noarr::idx<'y'>(40) + noarr::idx<'z'>(30));
	STATIC_REQUIRE(s2 + s3 == noarr::idx<'x'>(20) + noarr::idx<'y'>(20) + noarr::idx<'z'>(60));

	STATIC_REQUIRE(s1 - s2 == -noarr::idx<'x'>(10) + noarr::idx<'y'>(20) - noarr::idx<'z'>(30));
	STATIC_REQUIRE(s1 - s3 == noarr::idx<'x'>(10) - noarr::idx<'y'>(0) + noarr::idx<'z'>(-30));
	STATIC_REQUIRE(s2 - s3 == noarr::idx<'x'>(20) - noarr::idx<'y'>(20) + noarr::idx<'z'>(0));
}
