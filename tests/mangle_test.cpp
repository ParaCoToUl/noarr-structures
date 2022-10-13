#include <catch2/catch.hpp>

#include "noarr/structures.hpp"
#include "noarr/structures/mangle.hpp"

#include <string>

template<char... Chars>
std::string cp2a(noarr::char_sequence<Chars...>) { return {Chars...}; }

static std::string int_name = "i" + std::to_string(sizeof(int) * 8);
static std::string size_name = "u" + std::to_string(sizeof(std::size_t) * 8);

TEST_CASE("Mangle scalar", "[mangle]") {
	using s = noarr::scalar<int>;
	using m = noarr::mangle<s>;
	REQUIRE(cp2a(m()) == "scalar<"+int_name+">");
}

TEST_CASE("Mangle array", "[mangle]") {
	using s = noarr::array<'x', 42, noarr::scalar<int>>;
	using m = noarr::mangle<s>;
	REQUIRE(cp2a(m()) == "array<'x',("+size_name+")42,scalar<"+int_name+">>");
}
