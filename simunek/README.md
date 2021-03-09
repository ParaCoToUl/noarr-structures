# Mother of all requiremets
https://github.com/krulis-martin/cuda-kmeans
https://www.ksi.mff.cuni.cz/teaching/projects-web/pravidla.pdf

# Main ideas & Core principles
### Real world case problem
- All GPU programs can be separated into: CPU logic/policies/GPU logic
- Policy = logic of data layout on GPU
- Data need to be send onto GPU in organized manner
- We can have huge data, which need to process.
- K-means as an example how complex it is

### Solution
- Create high performace core for data modeling
- This framework will handle data transfers
- Library of common policies + extendability
- Support for data serialization, lifning and streaming

### Prove of usability
- Create binding into common languages like Python/R
- Rewrite cuda-kmeans using new framework, to gain following:
  - Demontrate usability of our libraly
  - Hopefully achive simpler and shorter code of k-means
  - Natural support of diffener data layouts (easy swaps) (AOS vs SOA)
  - Similar or higher level of performance

### After effects
- Add fancy formating to this document
- Get somehow at last mark 3 after we fail to achieve anything.
- Run.



# Requiremets
**Functional requiremets**
- Support for C++, bindings to Python, R
- Predefined basic data layouts for quick ot of the box usage.

**Quality requiremets**
 - Dots syntax








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
