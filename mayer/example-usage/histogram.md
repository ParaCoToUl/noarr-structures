# Image histogram

## The task

We have a grayscale `char[][]` image of width `w` and height `h`. Its size is dynamic. We want to compute a frequency table for each pixel intensity `int[256]`.


## Solution overview

We have a `BLOCK_SIZE = 1024` constant that specifies how many threads can run in parallel.

1. Interpret the image as a 1D list of `char` values.
2. Split the list into chunks of a size that fits into one thread block.
3. Compute a histogram for each chunk.
4. Run 256 threads to sum all the histograms into the final one.


## External program API

```python
# Python

import numpy as np

def compute_histogram(img: np.array) -> np.array:
    assert len(img.shape) == 2
    assert img.dtype == np.uint8
    
    histogram = # call c++ here
    assert len(histogram.shape) == 1
    assert histogram.dtype == np.int32
    
    return histogram
```

```cpp
// C++

void compute_histogram(
    const char *pixels,
    const std::size_t width,
    const std::size_t height,
    std::array<int, 256> &histogram
) {
    // call implementation here
}
```


## Example implementation without Noarr

```cpp
__global__ void reduce_chunks(char *pixels, int chunkSize, int *chunkHistograms) {
    int chunkIndex = threadIdx.x;
    int chunkStart = chunkIndex * chunkSize;

    char *chunk = pixels + chunkStart;
    int *histogram = chunkHistograms + chunkIndex * 256;
    
    // clear the chunk histogram
    for (int i = 0; i < 256; i++) {
        histogram[i] = 0;
    }

    // populate the chunk histogram by running over the chunk data
    for (int i = 0; i < chunkSize; i++) {
        histogram[chunk[i]] += 1;
    }
}

__global__ void reduce_histograms(int *chunkHistograms, int histogramCount, int *finalHistogram) {
    int histogramBin = threadIdx.x;

    finalHistogram[histogramBin] = 0;

    // sum one bin over all chunks
    for (int i = 0; i < histogramCount; i++) {
        finalHistogram[histogramBin] += chunkHistograms[256 * i + histogramBin];
    }
}

void compute_histogram(
    const char* pixels,
    const std::size_t width,
    const std::size_t height,
    std::array<int, 256>& histogram
) {
    const std::size_t BLOCK_SIZE = 1024; // max threads per block

    std::size_t pixelCount = width * height;
    std::size_t chunkSize = (pixelCount / BLOCK_SIZE) + 1;
    std::size_t chunkCount = BLOCK_SIZE;

    // move data onto the device & allocate buffers (pseudocode)
    char *d_pixels = copy(pixels);
    int *d_chunkHistograms = allocate(sizeof(int) * 256 * chunkCount);
    int *d_finalHistogram = allocate(sizeof(int) * 256);

    // run kernels
    reduce_chunks<<<1, chunkCount>>>(d_pixels, chunkSize, d_chunkHistograms);
    reduce_histograms<<<1, 256>>>(d_chunkHistograms, chunkCount, d_finalHistogram);

    // get data out of the device
    histogram = move(d_finalHistogram); // *pseudo code

    // free up device buffers
    free(d_pixels);
    free(d_chunkHistograms);
    free(d_finalHistogram);
}
```


## Example implementation with Noarr

```cpp
template<typename SP, typename SH>
__global__ void reduce_chunks(
    char *pixels, SP s_pixels,
    int *chunkHistograms, SH s_chunkHistograms
) {
    int chunkIndex = threadIdx.x;
    int chunkSize = s_pixels | noarr::length<'p'>();

    auto chunk = s_pixels | noarr::fix<'c'>(chunkIndex);
    auto histogram = s_chunkHistograms | noarr::fix<'c'>(chunkIndex);
    
    // clear the chunk histogram
    for (int i = 0; i < 256; i++) {
        histogram | noarr::fix<'b'>(i) | noarr::get_at(chunkHistograms) = 0;
    }

    // populate the chunk histogram by running over the chunk data
    for (int i = 0; i < chunkSize; i++) {
        histogram | noarr::fix<'b'>(
            chunk | noarr::fix<'p'>(i) | noarr::get_at(pixels)
        ) | noarr::get_at(chunkHistograms) += 1;
    }
}

template<typename SH>
__global__ void reduce_histograms(
    int *chunkHistograms, SH s_chunkHistograms,
    int *finalHistogram
) {
    int histogramBin = threadIdx.x;
    int histogramCount = s_chunkHistograms | noarr::length<'c'>();

    finalHistogram[histogramBin] = 0;

    // sum one bin over all chunks
    for (int c = 0; c < histogramCount; c++) {
        finalHistogram[histogramBin] += s_chunkHistograms
            | noarr::fixs<'c', 'b'>(c, histogramBin)
            | noarr::get_at(chunkHistograms);
    }
}

void compute_histogram(
    const char* pixels,
    const std::size_t width,
    const std::size_t height,
    std::array<int, 256>& histogram
) {
    const std::size_t BLOCK_SIZE = 1024; // max threads per block

    std::size_t pixelCount = width * height;
    std::size_t chunkSize = (pixelCount / BLOCK_SIZE) + 1;
    std::size_t chunkCount = BLOCK_SIZE;

    // re-interpret the given blob with noir as a sequence of chunks
    // c = chunk
    // p = pixel
    auto s_pixels = noarr::vector<'c', noarr::vector<'p', noarr::scalar<char>>>()
        | noarr::resize<'c'>(chunkCount)
        | noarr::resize<'p'>(chunkSize);

    // create structure for chunk histograms
    // c = chunk
    // b = bin
    auto s_chunkHistograms = noarr::vector<'c', noarr::array<'b', 256, noarr::scalar<int>>>()
        | noarr::resize<'c'>(chunkCount);

    // move data onto the device & allocate buffers (pseudocode)
    char *d_pixels = copy(pixels);
    int *d_chunkHistograms = allocate(s_chunkHistograms | noarr::get_size());
    int *d_finalHistogram = allocate(sizeof(int) * 256);

    // run kernels
    reduce_chunks<<<1, chunkCount>>>(
        d_pixels, s_pixels,
        d_chunkHistograms, s_chunkHistograms
    );
    reduce_histograms<<<1, 256>>>(
        d_chunkHistograms, s_chunkHistograms,
        d_finalHistogram
    );

    // get data out of the device
    histogram = move(d_finalHistogram); // *pseudo code

    // free up device buffers
    free(d_pixels);
    free(d_chunkHistograms);
    free(d_finalHistogram);
}
```


## Hypothetical dreamland API

Design a wrapper that will hold the structure, allow dot access and hold the host blob and the device blob.

To which blob we're accessing can be toggled by some `.toDevice()` or `.toHost()` methods. Etc...

Methods on the wrapper:

> **Question:** Why do we have unsized vector, when a vector sized to 0 would suffice?

> **Performed renaming:**
> - `resize` to `set_length` for consistency with `length`
> - `get_size` to `blob_size` to be explicit what size is meant

- setting sizes
    - `set_length<'x'>(4)` set runtime size along a dimension
    - `set_length<'x', 4>()` set constant size along a dimension
- querying sizes
    - `length<'x'>()` number of items along a dimensions
    - `blob_size()` size of the underlying blob in bytes
- accessing elements
    - `fix<'x'>(x)` fix an index along a dimension
    - `fixs<'x', 'y'>(x, y)` fix multiple dimensions simultaneously
    - `get()` get reference to the target scalar (all dimensions need be fixed)
- blob control
    - `set_host_blob(void* blob)`
    - `set_device_blob(void* blob)`
    - `allocate_host_blob()`
    - `allocalte_device_blob()`
    - `free_blobs()`
- blob access context
    - `to_device()` ... TODO: figure out better names here (meaning + usage context)
    - `to_host()`

```cpp
template<typename TPixels, typename TChunkHistograms>
__global__ void reduce_chunks(
    TPixels pixels,
    TChunkHistograms chunkHistograms
) {
    int chunkIndex = threadIdx.x;
    int chunkSize = pixels.length<'p'>();

    auto chunk = pixels.fix<'c'>(chunkIndex);
    auto histogram = chunkHistograms.fix<'c'>(chunkIndex);
    
    // clear the chunk histogram
    for (int i = 0; i < 256; i++) {
        histogram.fix<'b'>(i).get() = 0;
    }

    // populate the chunk histogram by running over the chunk data
    for (int i = 0; i < chunkSize; i++) {
        histogram.fix<'b'>(
            chunk.fix<'p'>(i).get()
        ).get() += 1;
    }
}

template<typename TChunkHistograms>
__global__ void reduce_histograms(
    TChunkHistograms chunkHistograms
    int *finalHistogram
) {
    int histogramBin = threadIdx.x;
    int histogramCount = chunkHistograms.length<'c'>();

    finalHistogram[histogramBin] = 0;

    // sum one bin over all chunks
    for (int c = 0; c < histogramCount; c++) {
        finalHistogram[histogramBin] += chunkHistograms
            .fixs<'c', 'b'>(c, histogramBin)
            .get();
    }
}

void compute_histogram(
    const char* pixels,
    const std::size_t width,
    const std::size_t height,
    std::array<int, 256>& histogram
) {
    const std::size_t BLOCK_SIZE = 1024; // max threads per block

    std::size_t pixelCount = width * height;
    std::size_t chunkSize = (pixelCount / BLOCK_SIZE) + 1;
    std::size_t chunkCount = BLOCK_SIZE;

    /**
     * TODO
     * 
     * POKRAČUJ TÍM, ŽE PŘEPÍŠEŠ ALOKACE BLOBŮ A BUDEŠ SPRÁVNĚ SWITCHOVAT KONTEXT
     */ 

    // re-interpret the given blob with noir as a sequence of chunks
    // c = chunk
    // p = pixel
    auto s_pixels = noarr::vector<'c', noarr::vector<'p', noarr::scalar<char>>>()
        .set_length<'c'>(chunkSize)
        .set_length<'p'>(chunkSize);

    // create structure for chunk histograms
    // c = chunk
    // b = bin
    auto s_chunkHistograms = noarr::vector<'c', noarr::array<'b', 256, noarr::scalar<int>>>()
        | noarr::resize<'c'>(chunkCount);

    // move data onto the device & allocate buffers (pseudocode)
    char *d_pixels = copy(pixels);
    int *d_chunkHistograms = allocate(s_chunkHistograms | noarr::get_size());
    int *d_finalHistogram = allocate(sizeof(int) * 256);

    // run kernels
    reduce_chunks<<<1, chunkCount>>>(
        d_pixels, s_pixels,
        d_chunkHistograms, s_chunkHistograms
    );
    reduce_histograms<<<1, 256>>>(
        d_chunkHistograms, s_chunkHistograms,
        d_finalHistogram
    );

    // get data out of the device
    histogram = move(d_finalHistogram); // *pseudo code

    // free up device buffers
    free(d_pixels);
    free(d_chunkHistograms);
    free(d_finalHistogram);
}
```
