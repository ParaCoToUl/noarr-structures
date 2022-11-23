#include <catch2/catch.hpp>

#include <noarr/structures.hpp>
#include <noarr/structures/extra/mangle.hpp>
#include <noarr/structures/structs/setters.hpp>

#include <string>
#include <stdint.h>

template<char... Chars>
std::string cp2a(noarr::char_sequence<Chars...>) { return {Chars...}; }

static std::string int_name = "int" + std::to_string(sizeof(int) * 8) + "_t";
static std::string size_name = "uint" + std::to_string(sizeof(std::size_t) * 8) + "_t";

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

TEST_CASE("Mangle expr", "[mangle]") {
	using namespace noarr;
	auto &sn = size_name; // shorten name for alignment

	auto structure =        set_length_t<'y',vector<'y',array<'x',(size_t)42,scalar<int32_t>>>,size_t>{vector<'y',array<'x',(size_t)42,scalar<int32_t>>>{array<'x',(size_t)42,scalar<int32_t>>{scalar<int32_t>{},},},size_t{24},};
	std::string expected = "set_length_t<'y',vector<'y',array<'x',("+sn+")42,scalar<int32_t>>>,"+sn+">{vector<'y',array<'x',("+sn+")42,scalar<int32_t>>>{array<'x',("+sn+")42,scalar<int32_t>>{scalar<int32_t>{},},},"+sn+"{24},}";
	std::string actual = noarr::mangle_expr<std::string>(structure);

	auto structure_pretty = noarr::array<'x', 42, noarr::scalar<int>>() ^ noarr::vector<'y'>() ^ noarr::set_length<'y'>(24);
	static_assert(std::is_same_v<decltype(structure), decltype(structure_pretty)>, "Test sanity check");

	REQUIRE(expected == actual);
}
