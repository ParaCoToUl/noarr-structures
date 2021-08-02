# Instalation & Usage
You can build the example using `CMake` in the same way as described in the main README. After compiling the example you will be able to run matrix example and a choose matrix layout and size dynamically.

# Implementation
In file [noarr_matrix_functions.hpp](noarr_matrix_functions.hpp) are implemented several matrix functions. The important function is:
```cpp
void matrix_multiply(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, noarr::bag<Structure3>& matrix3)
```
it is given 3 matrices and it multiplies first two into third one.

In file [matrix.cpp](matrix.cpp) basic matrix is implemented. Example first generates 2 classic matrices. Then it copies them into a noarr version. The example then performs multiplications separately. It then copies the noarr result into a normal version and compares the results if they are equal.

You are able to choose from several layouts. The first two are modeled using basic `noarr` features. The third one is using `z_curve` implemented in [z_curve.cpp](z_curve.cpp).
