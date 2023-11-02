#ifndef NOARR_TEST_COUNTER_HPP
#define NOARR_TEST_COUNTER_HPP

namespace noarr_test {

#include <exception>

struct test_counter_t {
	static inline int test_failure = 0;
	static inline int test_success = 0;
	static inline int assertion_failure = 0;
	static inline int assertion_success = 0;
};

inline void test_failed() {
	++test_counter_t::test_failure;
	++test_counter_t::test_success;
}

inline void test_passed() {
	++test_counter_t::test_success;
}

inline void assertion_failed() {
	++test_counter_t::assertion_failure;
	++test_counter_t::assertion_success;
}

inline void assertion_passed() {
	++test_counter_t::assertion_success;
}

struct requirement_failed : std::exception {};

}

#endif // NOARR_TEST_COUNTER_HPP
