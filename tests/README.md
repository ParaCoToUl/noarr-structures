# Noarr Structures tests outline

- [compile_test.cpp](compile_test.cpp): contains the code snippets from the root [README.md](../README.md).
- [containers_dots_test.cpp](containers_dots_test.cpp): checks that the `set_length` method correctly sets the lengths of wrapped vectors and arrays
- [containers_pipes_test.cpp](containers_pipes_test.cpp): checks that the `set_length` function correctly sets the lengths of unwrapped vectors and arrays (via the piping mechanism)
- [histogram_test.cpp](histogram_test.cpp): contains a simple histogram computation on an image and checks its correctness
- [image_test.cpp](image_test.cpp): <!-- TODO -->
- [literal_test.cpp](literal_test.cpp): checks that the provided structures and their sizes can be created/computed during compile time
- [matrix_test.cpp](matrix_test.cpp): contains a few example computations with matrices and checks their correctness
- [pod_test.cpp](pod_test.cpp): checks that the provided structures are trivial standard layouts
- [reassemble_test.cpp](reassemble_test.cpp): checks the correctness of the `reassemble` function
- [size_dots_test.cpp](size_dots_test.cpp): checks that the sizes of structures are correctly computed after setting lengths of arrays and vectors contained in them (using the dot notation)
- [size_pipes_test.cpp](size_pipes_test.cpp): checks that the sizes of structures are correctly computed after setting lengths of arrays and vectors contained in them (using the `|` notation)
