cmake_minimum_required(VERSION 3.10)
Include(FetchContent)

# set the project name
project(NoarrStructuresMatrixExample VERSION 0.1)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# use threading library
find_package(Threads REQUIRED)

# setup the test runner executable
add_executable(matrix matrix.cpp)
target_include_directories(matrix PUBLIC ../../include)
target_link_libraries(matrix PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(matrix PRIVATE /W4)
else()
  target_compile_options(matrix PRIVATE -Wall -Wextra -pedantic)
endif()
