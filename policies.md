# Policies

- policy classes on data layouts (tuples, arrays, vectors, primitive types; combinations of such)
- main policies: (all of those ~fit in a `std::array<float, 3 * N>`)
  - SoA

    ```cpp
    struct point_list3D {
      float x[N];
      float y[N];
      float z[N];
    };
    struct point_list3D points;
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

## requirements

1. give me an array of pixels (e.g. `tuple<float,int,char>`) = AoS
2. here is a pixel (e.g. `tuple<float,int,char>`), make a SoA representation of it
3. it should be quite easy for Aos and SoA
4. we want to be able to enforce alignment
5. it has to provide the `at` (get by index) function
6. automatic definitions of iterators would be nice

## How to do it + ideas

- we want something like a primitive type system (no functions, no pointers, just raw data)
