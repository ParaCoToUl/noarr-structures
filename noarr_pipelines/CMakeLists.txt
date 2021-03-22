cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(Noarr VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# download testing library Catch2
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1)
FetchContent_MakeAvailable(Catch2)

# setup the test runner executable
file(GLOB tests_source_files CONFIGURE_DEPENDS
    "tests/*_test.hpp"
    "tests/*_test.cpp"
    "tests/**/*_test.hpp"
    "tests/**/*_test.cpp"
)
add_executable(test-runner tests/main_test.cpp ${tests_source_files})
target_include_directories(test-runner PUBLIC src)
target_link_libraries(test-runner PRIVATE Catch2::Catch2)

# setup CTest integration
file(
    DOWNLOAD
    "https://raw.githubusercontent.com/catchorg/Catch2/devel/extras/ParseAndAddCatchTests.cmake"
    "ParseAndAddCatchTests.cmake")
enable_testing()
cmake_policy(SET CMP0110 OLD)
include("ParseAndAddCatchTests.cmake")
ParseAndAddCatchTests(test-runner)
