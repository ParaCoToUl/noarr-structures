# Policies

- policy classes on data layouts (tuples, arrays, vectors, primitive types; combinations of such)
- main policies: (all of those ~fit in a `std::arrray<float, 3 * N>`)
  - SoA
    ```cpp
    struct pointlist3D {
      float x[N];
      float y[N];
      float z[N];
    };
    struct pointlist3D points;
    float get_point_x(int i) { return points.x[i]; }
    ```
  - AoS
    ```cpp
    struct point3D {
        float x;
        float y;
        float z;
    };
    struct point3D points[N];
    float get_point_x(int i) { return points[i].x; }
    ```
  - AoSoA (great for simd processing)
    ```cpp
    struct point3Dx8 {
        float x[8];
        float y[8];
        float z[8];
    };
    struct point3Dx8 points[(N+7)/8];
    float get_point_x(int i) { return points[i/8].x[i%8]; }
    ```

## How to do it? + ideas

- we want to tell the `std::array<float, 3 * N>` what its data are like (e.g. one of the above)
- we would like it to know that `SoA std::array<float, 3 * N> == std::tuple<std::array<float, N>, std::array<float, N>, std::array<float, N>>` (supposing there is no alignment involved) etc.
- we want to be able to enforce an alignment
- do we want something like a primitive type system (no functions, no pointers, just raw data)? (open for discussion)
- we would like it if it could provide us the get_by_index functions

### Possible strategies

- variadic templates:
  - first idea
    ```cpp
    tuple<array<float, N, primitive>, array<float, N, primitive>, array<float, N, primitive>> = array<float, 3 * N, SoA<N,N,N>>
    ```
    - Pak na kazdem specialni `get` funkce (tak to maj i tuply)
      ```cpp
      T &get<0, array<T,N,primitive>>(array &a, size_t index) { return a[index]; }
      T &get<n, array<T,N,SoA<N,N,N>>>(array &a, size_t index) { return a[N*n + index]; }
      ```
### Streaming vs One-chunk approach
