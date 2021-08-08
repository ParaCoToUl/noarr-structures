# Noarr

This folder contains the the Noarr library.

To build the test project on linux:

```sh
# creates the build directory
cmake -E make_directory build

# enters the build directory
cd build

# configures the build environment
cmake ..

# builds the project according to the configuration
cmake --build .
```

Then, to run tests on linux/macos:

```sh
./test-runner
```

To build and test on windows, run CMake the same way from cygwin or gitbash.
