# Mother of all requiremets
- https://github.com/krulis-martin/cuda-kmeans
- https://www.ksi.mff.cuni.cz/teaching/projects-web/pravidla.pdf
- https://github.com/jiriklepl/SWP-PLAYGROUND/blob/master/simunek/Z%C3%A1m%C4%9Br%20t%C3%BDmov%C3%A9ho%20projektu.pdf

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

### Features / Timeline
- noarr - high performance indexing structure?
- Blobs
- Data serialization, lifting
- Data streaming/uploads: pipelines - envelopes, CPU/GPU nodes, links

### k-means™ powered by noarr™
- Code size
- Speed
- Supported layouts

### After effects
- Add fancy formating to this document
- Get somehow at last mark 3 after we fail to achieve anything.
- Run.
