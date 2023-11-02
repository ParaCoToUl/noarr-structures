#ifndef MSVC_TESTER_CATCH2_FAILURE_COUNTER_HPP
#define MSVC_TESTER_CATCH2_FAILURE_COUNTER_HPP

#include <stdexcept>

struct test_counter_t {
	static int failures;
	static int tests;
};

inline int test_counter_t::failures = 0;
inline int test_counter_t::tests = 0;

inline void test_failed() {
	++test_counter_t::failures;
	++test_counter_t::tests;
}

inline void test_passed() {
	++test_counter_t::tests;
}

struct requirement_failed : std::exception {};


#endif // MSVC_TESTER_CATCH2_FAILURE_COUNTER_HPP
