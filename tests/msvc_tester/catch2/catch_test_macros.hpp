#ifndef MSVC_TESTER_CATCH2_CATCH_TEST_MACROS_HPP
#define MSVC_TESTER_CATCH2_CATCH_TEST_MACROS_HPP

#include <iostream>

#include "msvc_tester_utils.hpp"

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)

#define TO_STRING_DETAIL(x) #x
#define TO_STRING(x) TO_STRING_DETAIL(x)

namespace {

bool test_case_failed = false;

}

#define REQUIRE(...) \
	do { \
		CHECK(__VA_ARGS__); \
		if (test_case_failed) throw requirement_failed{}; \
	} while (false)

#define CHECK(...) \
	do if (!(__VA_ARGS__)) { \
		std::cerr << __FILE__ "(" TO_STRING(__LINE__) "): error: requirement \"" #__VA_ARGS__  "\"  failed" << std::endl; \
		test_case_failed = true; \
	} while (false)

#define SECTION(name) \
	struct CONCATENATE(test_case_section_, __LINE__) { \
		~CONCATENATE(test_case_section_, __LINE__)() { \
			if (test_case_failed) \
				std::cerr << __FILE__ "(" TO_STRING(__LINE__) "): message: in section \"" name "\"" << std::endl; \
		} \
	}; \
	if (CONCATENATE(test_case_section_, __LINE__) CONCATENATE(test_case_section_instance_, __LINE__); false) ; else

#define TEST_CASE(name, tags) \
	static void CONCATENATE(test_case_function_, __LINE__)(); \
	namespace { \
	struct CONCATENATE(test_case_struct_, __LINE__) { \
		CONCATENATE(test_case_struct_, __LINE__)() { \
			test_case_failed = false; \
			try { CONCATENATE(test_case_function_, __LINE__)(); } \
			catch (const requirement_failed&) { test_case_failed = true; } \
			if (test_case_failed) { \
				std::cerr << __FILE__ "(" TO_STRING(__LINE__) "): message: in test_case \"" name "\" " tags "" << std::endl; \
				test_failed(); \
			} else { \
				test_passed(); \
			} \
		} \
	} CONCATENATE(test_case_instance_, __LINE__); \
	} \
	static void CONCATENATE(test_case_function_, __LINE__)()

#endif // MSVC_TESTER_CATCH2_CATCH_TEST_MACROS_HPP
