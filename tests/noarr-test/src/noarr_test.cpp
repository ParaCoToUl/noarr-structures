#include <iostream>

#include "noarr_test/counter.hpp"

using namespace noarr_test;

int main() {
	std::cerr << "Tests: " << test_counter_t::test_success << "\n";
	std::cerr << "Failures: " << test_counter_t::test_failure << "\n";

	std::cerr << "Assertions: " << test_counter_t::assertion_success << "\n";
	std::cerr << "Failures: " << test_counter_t::assertion_failure << "\n";

	if (test_counter_t::test_failure > 0) {
		return 1;
	}
}
