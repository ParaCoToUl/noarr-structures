# cuda_striped

Create multiple interleaved copies (stripes) of a structure for use with CUDA shared memory.

```hpp
#include <noarr/structures/interop/cuda_striped.hpp>

template<std::size_t NumStripes, std::size_t BankCount = 32, std::size_t BankWidth = 4>
constexpr proto noarr::cuda_striped();

template<std::size_t NumStripes, typename ElemType, std::size_t BankCount = 32, std::size_t BankWidth = 4>
constexpr proto noarr::cuda_striped();
```

(`proto` is an unspecified [proto-structure](../Glossary.md#proto-structure))


## Description

These two functions return a proto-structure that transforms a structure into several copies (stripes), interleaved to allow efficient usage of shared memory.

The new structure does *not* contain the original structure's layout. Instead, the original layout is split into `ElemType` pieces.
`ElemType` must be a noarr structure type. If it is not specified, it defaults to the appropriate [`noarr::scalar`](scalar.md).
Only the inner content of each `ElemType` piece is stored consecutively.

The elements of the structure (and their [offets](../Glossary.md#offset)) can only be accessed in device code
(although the [size](../Glossary.md#size) and other properties can be queried anywhere -- host or device code).
The stripe to be accessed is selected automatically to be `threadIdx.x % NumStripes`.
Alternatively, you can optionally select the copy explicitly by adding a `noarr::cuda_stripe_index` to the [state](../State.md) used during the query.

The exact layout is unspecified, but the following properties are guaranteed:
- no 4-byte (more precisely: `BankWidth`-byte) memory cell will contain data of two different stripes
- if possible, no bank will contain data of two different stripes (otherwise, the number of such bank conflicts are minimized)
- each stripe has the same memory layout and the offset between neighboring stripes is constant (but it may differ from the offset between the last stripe and the first one)

There may be some padding involved, do not assume that the size of the structure will be the original structure size multiplied by the number of stripes.
Also, do not assume that the offset between individual elements within one stripe is constant.
