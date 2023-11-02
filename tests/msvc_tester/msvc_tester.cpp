#include <iostream>

#include "catch2/msvc_tester_utils.hpp"

int main() {
	std::cout << "Tests: " << test_counter_t::tests << std::endl;
	std::cout << "Failures: " << test_counter_t::failures << std::endl;

	if (test_counter_t::failures > 0) {
		return 1;
	}
}
