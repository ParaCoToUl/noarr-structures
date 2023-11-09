cmake_minimum_required(VERSION 3.10)

# set the project name
project(
  NoarrTests VERSION 0.20.0
  DESCRIPTION "Tests for the Noarr library"
  LANGUAGES CXX
)

include(CTest)
enable_testing()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# setup the test runner executable
file(GLOB_RECURSE tests_source_files CONFIGURE_DEPENDS
    "*_test.cpp"
)

add_library(noarr_test STATIC noarr-test/src/noarr_test.cpp)
target_include_directories(noarr_test PUBLIC noarr-test/include)

add_executable(test-runner ${tests_source_files})
target_include_directories(test-runner PRIVATE ../include)

target_link_libraries(test-runner PRIVATE noarr_test)

# ask the compiler to print maximum warnings
if(MSVC)
  target_compile_options(test-runner PRIVATE /W4)
else()
  target_compile_options(test-runner PRIVATE -Wall -Wextra -Wno-parentheses -pedantic)
endif()

add_test(NAME test-runner COMMAND test-runner)
