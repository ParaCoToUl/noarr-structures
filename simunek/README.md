# Mother of all requiremets
https://github.com/krulis-martin/cuda-kmeans

# Main ideas & Core principles
1. All GPU programs can be separated into: CPU logic/policies/GPU logic
2. Policy = logic of data transfering and accesing them on GPU.
3. Create high performace core for data modeling
4. Create library of common policies
5. Create binding into common languages like Python/R
6. Rewrite cuda-kmeans using new framework, to gain following:
  a. Demontrate usability of our libraly
  b. Hopefully achive simpler and shorter code of k-means
  c. Natural support of diffener data layouts (easy swaps) (AOS vs SOA)
  d. Similar or higher level of performance
7. Add fancy formating to this document
8. Get somehow at last mark 3 after we fail to achieve anything.
9. Run.

# Requiremets
**Functional requiremets**
- Support for C++, bindings to Python, R
- Predefined basic data layouts for quick ot of the box usage.

**Quality requiremets**
Usability
- dots








# Jirka wrote earlier (To be depricated):
Separate data layout from data access.
    Why: Algorithm implemementation should not depend on data layout.

Make the tool easy to use.
    => automatic return value typing
    => passing data slices to functions (to aid abstraction)
        HINT: define virtual data layouts that map to physical data layout?

Make the tool performant.
    => use compile-time features as much as possible
    => aviod pointer derefecencing when possible


Observations
------------

Tuples should not be iterated at runtime, causes return value ambiguity.
    RESULT: No, they have to be, because paths have to be runtime only.
    Otherwise we need to pass compiletime values as template arguments EVERYWHERE,
    even in the user code which makes it super messy.
    
    But that's ok, coz we infer the value type from virtual layout.

All algorithms at some point work with scalar values only - make scalar value access the priority.
