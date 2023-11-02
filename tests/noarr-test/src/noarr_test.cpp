#include <iostream>

#include "noarr_test/counter.hpp"

using namespace noarr_test;

int main() {
	std::cerr << "Tests: " << test_counter_t::test_success << std::endl;
	std::cerr << "Failures: " << test_counter_t::test_failure << std::endl;

	std::cerr << std::endl;

	std::cerr << "Assertions: " << test_counter_t::assertion_success << std::endl;
	std::cerr << "Failures: " << test_counter_t::assertion_failure << std::endl;

	if (test_counter_t::test_failure > 0) {
		return 1;
	}
}
