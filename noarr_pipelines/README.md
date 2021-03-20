# Noarr (pipelines)

This folder contains the the Noarr library.

> **NOTE:** It contains only the `pipelines` module, the `structures` module will be migrated in here later. (Why here and not the other way around? Because we want to migrate to CMake due to platform independence.)

To build the project on linux:

    cmake --build .

To run tests on linux:

    ctest

To build on windows, run CMake and dump created files into a `/out` folder as it's already added to the `.gitignore` file. Then use visual studio.
