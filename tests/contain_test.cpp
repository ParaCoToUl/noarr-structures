#include <noarr_test/macros.hpp>
#include <noarr_test/defs.hpp>

#include <noarr/structures_extended.hpp>

using namespace noarr;

TEST_CASE("Empty contain test") {
	struct empty {};
	constexpr auto contain = strict_contain<>();
	STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
	STATIC_REQUIRE(sizeof(contain) == sizeof(empty));
}

TEST_CASE("Atomic contain test") {
	{
		constexpr auto contain = strict_contain<char>('a');
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(char));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		REQUIRE(a == 'a');
		REQUIRE(&a == &contain.get<0>());
	}

	{
		constexpr auto contain = strict_contain<int>(1);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(int));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const int&>);
		REQUIRE(a == 1);
		REQUIRE(&a == &contain.get<0>());
	}

	{
		constexpr auto contain = strict_contain<float>(2.0f);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(float));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const float&>);
		REQUIRE(!(a < 2.0f) && !(2.0f < a));
		REQUIRE(&a == &contain.get<0>());
	}

	{
		constexpr auto contain = strict_contain<double>(3.0);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(double));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const double&>);
		REQUIRE(!(a < 3.0) && !(3.0 < a));
		REQUIRE(&a == &contain.get<0>());
	}

	{
		constexpr auto contain = strict_contain<void*>(nullptr);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(void*));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), void*const&>);
		REQUIRE(a == nullptr);
		REQUIRE(&a == &contain.get<0>());
	}

	{
		constexpr auto contain = strict_contain<char*>(nullptr);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(char*));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), char*const&>);
		REQUIRE(a == nullptr);
		REQUIRE(&a == &contain.get<0>());
	}

	{
		struct empty {};
		constexpr auto contain = strict_contain<empty>(empty{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(empty));

		const auto &[a] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const empty>);
		static_cast<void>(a);
	}
}

TEST_CASE("Pair contain test") {
	{
		struct pair {
			char a;
			int b;
		};

		constexpr auto contain = strict_contain<char, int>('a', 1);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(pair));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(pair));

		const auto &[a, b] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const int&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 1);
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
	}

	{
		struct pair {
			int a;
			char b;
		};

		constexpr auto contain = strict_contain<int, char>(1, 'a');
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(pair));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(pair));

		const auto &[a, b] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const int&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		REQUIRE(a == 1);
		REQUIRE(b == 'a');
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
	}

	{
		struct pair {
			float a;
			double b;
		};

		constexpr auto contain = strict_contain<float, double>(2.0f, 3.0);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(pair));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(pair));

		const auto &[a, b] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const float&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const double&>);
		REQUIRE(!(a < 2.0f) && !(2.0f < a));
		REQUIRE(!(b < 3.0) && !(3.0 < b));
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
	}

	{
		struct pair {
			void* a;
			char* b;
		};

		constexpr auto contain = strict_contain<void*, char*>(nullptr, nullptr);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(pair));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(pair));

		const auto &[a, b] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), void*const&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), char*const&>);
		REQUIRE(a == nullptr);
		REQUIRE(b == nullptr);
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
	}

	{
		struct pair {
			char a;
			char b;
		};

		constexpr auto contain = strict_contain<char, char>('a', 'b');
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(pair));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(pair));

		const auto &[a, b] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 'b');
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
	}
}

TEST_CASE("Triple contain test") {
	{
		struct triple {
			char a;
			int b;
			float c;
		};

		constexpr auto contain = strict_contain<char, int, float>('a', 1, 2.0f);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[a, b, c] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const int&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const float&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 1);
		REQUIRE(!(c < 2.0f) && !(2.0f < c));
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
		REQUIRE(&c == &contain.get<2>());
	}

	{
		struct triple {
			int a;
			char b;
			float c;
		};

		constexpr auto contain = strict_contain<int, char, float>(1, 'a', 2.0f);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[a, b, c] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const int&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const float&>);
		REQUIRE(a == 1);
		REQUIRE(b == 'a');
		REQUIRE(!(c < 2.0f) && !(2.0f < c));
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
		REQUIRE(&c == &contain.get<2>());
	}

	{
		struct triple {
			float a;
			double b;
			char c;
		};

		constexpr auto contain = strict_contain<float, double, char>(2.0f, 3.0, 'a');
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[a, b, c] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const float&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const double&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const char&>);
		REQUIRE(!(a < 2.0f) && !(2.0f < a));
		REQUIRE(!(b < 3.0) && !(3.0 < b));
		REQUIRE(c == 'a');
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
		REQUIRE(&c == &contain.get<2>());
	}

	{
		struct triple {
			void* a;
			char* b;
			int* c;
		};

		constexpr auto contain = strict_contain<void*, char*, int*>(nullptr, nullptr, nullptr);
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[a, b, c] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), void*const&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), char*const&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), int*const&>);
		REQUIRE(a == nullptr);
		REQUIRE(b == nullptr);
		REQUIRE(c == nullptr);
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
		REQUIRE(&c == &contain.get<2>());
	}

	{
		struct triple {
			char a;
			char b;
			char c;
		};

		constexpr auto contain = strict_contain<char, char, char>('a', 'b', 'c');
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[a, b, c] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const char&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 'b');
		REQUIRE(c == 'c');
		REQUIRE(&a == &contain.get<0>());
		REQUIRE(&b == &contain.get<1>());
		REQUIRE(&c == &contain.get<2>());
	}
}

TEST_CASE("Triple contain tests with empty between items") {
	struct empty_t {};

	{
		struct triple {
			char a;
			int b;
			float c;
		};

		constexpr auto contain = strict_contain<empty_t, char, empty_t, int, empty_t, float, empty_t>
			(empty_t{}, 'a', empty_t{}, 1, empty_t{}, 2.0f, empty_t{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[e1, a, e2, b, e3, c, e4] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const int&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const float&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 1);
		REQUIRE(!(c < 2.0f) && !(2.0f < c));
		REQUIRE(&a == &contain.get<1>());
		REQUIRE(&b == &contain.get<3>());
		REQUIRE(&c == &contain.get<5>());
	}

	{
		struct triple {
			int a;
			char b;
			float c;
		};

		constexpr auto contain = strict_contain<empty_t, int, empty_t, char, empty_t, float, empty_t>
			(empty_t{}, 1, empty_t{}, 'a', empty_t{}, 2.0f, empty_t{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[e1, a, e2, b, e3, c, e4] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const int&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const float&>);
		REQUIRE(a == 1);
		REQUIRE(b == 'a');
		REQUIRE(!(c < 2.0f) && !(2.0f < c));
		REQUIRE(&a == &contain.get<1>());
		REQUIRE(&b == &contain.get<3>());
		REQUIRE(&c == &contain.get<5>());
	}

	{
		struct triple {
			float a;
			double b;
			char c;
		};

		constexpr auto contain = strict_contain<empty_t, float, empty_t, double, empty_t, char, empty_t>
			(empty_t{}, 2.0f, empty_t{}, 3.0, empty_t{}, 'a', empty_t{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[e1, a, e2, b, e3, c, e4] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const float&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const double&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const char&>);
		REQUIRE(!(a < 2.0f) && !(2.0f < a));
		REQUIRE(!(b < 3.0) && !(3.0 < b));
		REQUIRE(c == 'a');
		REQUIRE(&a == &contain.get<1>());
		REQUIRE(&b == &contain.get<3>());
		REQUIRE(&c == &contain.get<5>());
	}

	{
		struct triple {
			void* a;
			char* b;
			int* c;
		};

		constexpr auto contain = strict_contain<empty_t, void*, empty_t, char*, empty_t, int*, empty_t>
			(empty_t{}, nullptr, empty_t{}, nullptr, empty_t{}, nullptr, empty_t{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[e1, a, e2, b, e3, c, e4] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), void*const&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), char*const&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), int*const&>);
		REQUIRE(a == nullptr);
		REQUIRE(b == nullptr);
		REQUIRE(c == nullptr);
		REQUIRE(&a == &contain.get<1>());
		REQUIRE(&b == &contain.get<3>());
		REQUIRE(&c == &contain.get<5>());
	}

	{
		struct triple {
			char a;
			char b;
			char c;
		};

		constexpr auto contain = strict_contain<empty_t, char, empty_t, char, empty_t, char, empty_t>
			(empty_t{}, 'a', empty_t{}, 'b', empty_t{}, 'c', empty_t{});
		STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
		STATIC_REQUIRE(sizeof(contain) == sizeof(triple));
		STATIC_REQUIRE(alignof(decltype(contain)) == alignof(triple));

		const auto &[e1, a, e2, b, e3, c, e4] = contain;
		STATIC_REQUIRE(std::is_same_v<decltype(a), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(b), const char&>);
		STATIC_REQUIRE(std::is_same_v<decltype(c), const char&>);
		REQUIRE(a == 'a');
		REQUIRE(b == 'b');
		REQUIRE(c == 'c');
		REQUIRE(&a == &contain.get<1>());
		REQUIRE(&b == &contain.get<3>());
		REQUIRE(&c == &contain.get<5>());
	}
}

TEST_CASE("Just empty") {
	struct empty {};
	constexpr auto contain = strict_contain<empty, empty, empty, empty>(empty{}, empty{}, empty{}, empty{});
	STATIC_REQUIRE(noarr_test::is_simple<decltype(contain)>);
	STATIC_REQUIRE(sizeof(contain) == sizeof(empty));
}
