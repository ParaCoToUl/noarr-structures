#### Matrix example tests
![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20matrix%20example%20test%20ubuntu-latest%20-%20clang/badge.svg)
![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20matrix%20example%20test%20ubuntu-latest%20-%20gcc/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20matrix%20example%20test%20macosl/badge.svg)

![CI status](https://github.com/ParaCoToUl/noarr/workflows/Noarr%20matrix%20example%20test%20Win/badge.svg)

# Instalation

TODO: Copy instructions from main README when ready!

You can build the example using `CMake` in the same way as described in the main README. After compiling the example you will be able to run matrix example and a choose matrix layout and size dynamically.

# Usage
```text
Programm takes 2 parameters. First, you choose one of the following layouts:
1) rows
2) columns
3) z_curve (the size has to be a power of 2)
Then you input integer matrix size. The size of the matrix have to be at least one. 
(for example simplicity, only square matrices are supported)
```

Running the example on Windows:
```text
.\matrix.exe rows 7
```

Running the example on Linux or Mac:
```text
./matrix columns 10
./matrix z_curve 8
```

# Implementation
Implementation is commented on in detail. We recommend starting reading [matrix.cpp](matrix.cpp), following with [noarr_matrix_functions.hpp](noarr_matrix_functions.hpp) and [z_curve.cpp](z_curve.cpp) last.

In file [matrix.cpp](matrix.cpp) basic matrix is implemented. Example first generates 2 classic matrices. Then it copies them into a noarr version. The example then performs multiplications separately. It then copies the noarr result into a normal version and compares the results if they are equal.

In file [noarr_matrix_functions.hpp](noarr_matrix_functions.hpp) are implemented several matrix functions. The important function is:
```cpp
noarr::bag<Structure3> noarr_matrix_multiply(noarr::bag<Structure1>& matrix1, noarr::bag<Structure2>& matrix2, Structure3 structure)
```
it is given 2 matrices and it multiplies first two into third one using `noarr`.

You are able to choose from several layouts. The first two are modeled using basic `noarr` features. The third one is using `z_curve` implemented in [z_curve.cpp](z_curve.cpp).
