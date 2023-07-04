cmake_minimum_required(VERSION 3.10)

project(Noarr
    VERSION 0.20.0
    LANGUAGES CXX
    DESCRIPTION "Noarr is a header-only library that facilitates the creation of many data structures and provides a layout-agnostic way of accessing the values stored in them."
)

add_library(Noarr INTERFACE)

set(NOARR_DIR ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(Noarr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

install(DIRECTORY include/ DESTINATION include)
