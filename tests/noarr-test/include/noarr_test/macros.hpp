#ifndef NOARR_TEST_MACROS_HPP
#define NOARR_TEST_MACROS_HPP

#include <iostream>
#include <type_traits>

#include "counter.hpp"

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

#define TO_STRING_DETAIL(x) #x
#define TO_STRING(x) TO_STRING_DETAIL(x)

namespace noarr_test {

namespace {

bool test_case_failed = false;

}

}

#define REQUIRE(...) \
	do if (!(__VA_ARGS__)) { \
		std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": error: REQUIRE(" #__VA_ARGS__  ") failed\n"; \
		noarr_test::assertion_failed(); \
		noarr_test::test_case_failed = true; \
		throw noarr_test::requirement_failed{}; \
	} else { \
		noarr_test::assertion_passed(); \
	} while (false)

#define STATIC_REQUIRE(...) \
	do { \
		const bool cond = [](auto cond) constexpr { return std::is_constant_evaluated() ? cond : true; }(!(__VA_ARGS__)); \
		if (cond) { \
			std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": error: STATIC_REQUIRE(" #__VA_ARGS__  ") failed\n"; \
			noarr_test::assertion_failed(); \
			noarr_test::test_case_failed = true; \
			throw noarr_test::requirement_failed{}; \
		} else { \
			noarr_test::assertion_passed(); \
		} \
	} while (false)

#define CHECK(...) \
	do if (!(__VA_ARGS__)) { \
		std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": error: CHECK(" #__VA_ARGS__  ") failed\n"; \
		noarr_test::assertion_failed(); \
		noarr_test::test_case_failed = true; \
	} else { \
		noarr_test::assertion_passed(); \
	} while (false)

#define SECTION(...) \
	if (struct CONCATENATE(test_case_section_, __LINE__) { \
		~CONCATENATE(test_case_section_, __LINE__)() { \
			if (noarr_test::test_case_failed) \
				std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": message: in SECTION(" #__VA_ARGS__ ")\n"; \
		} \
	} CONCATENATE(test_case_section_instance_, __LINE__); false) ; else

#define TEST_CASE(...) \
	static void CONCATENATE(test_case_function_, __LINE__)(); \
	namespace { \
	const struct CONCATENATE(test_case_struct_, __LINE__) { \
		CONCATENATE(test_case_struct_, __LINE__)() { \
			noarr_test::test_case_failed = false; \
			try { CONCATENATE(test_case_function_, __LINE__)(); } \
			catch (const noarr_test::requirement_failed&) { noarr_test::test_case_failed = true; } \
			catch (...) { \
				std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": error: unexpected exception\n"; \
				noarr_test::test_case_failed = true; \
			} \
			if (noarr_test::test_case_failed) { \
				std::cerr << __FILE__ ":" TO_STRING(__LINE__) ": message: in TEST_CASE(" #__VA_ARGS__ ")\n"; \
				noarr_test::test_failed(); \
			} else { \
				noarr_test::test_passed(); \
			} \
		} \
	} CONCATENATE(test_case_instance_, __LINE__); \
	} \
	static void CONCATENATE(test_case_function_, __LINE__)()

#endif // NOARR_TEST_MACROS_HPP
