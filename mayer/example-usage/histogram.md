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
    cudaMemcpy(histogram, d_finalHistogram, sizeof(d_finalHistogram), cudaMemcpyDeviceToHost);

    // free up device buffers
    cudaFree(d_pixels);
    cudaFree(d_chunkHistograms);
    cudaFree(d_finalHistogram);
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
    cudaMemcpy(histogram, d_finalHistogram, sizeof(d_finalHistogram), cudaMemcpyDeviceToHost);

    // free up device buffers
    cudaFree(d_pixels);
    cudaFree(d_chunkHistograms);
    cudaFree(d_finalHistogram);
}
```


## Hypothetical dreamland API

Design a wrapper that will hold the structure, allow dot access and hold the host blob and the device blob.

To which blob we're accessing can be toggled by some `.toDevice()` or `.toHost()` methods. Etc...

Methods on the wrapper:

> **Question:** Why do we have unsized vector, when a vector sized to 0 would suffice?

**API Reference:**

- `noarr::structure<T>`
    - `.set_length<'x'>(4)` set size along a dimensions
    - `.set_length<'x', 4>()` set constant size along a dimensions
    - `.length<'x'>()` get number of items along a dimension
    - `.blob_size()` size of the underlying blob in bytes
    - `.fix<'x'>(x)` fix an index along a dimension
    - ??? tuple fixing? (`.fixt<'x', 2>()`?)
    - `.blob_offset()` byte offset of the fixed item in the blob
    - `.get_at(void* blob)` get fixed item inside a blob
    - `.with_blob(blob | void*)` create a `structure_with_blob`
- `noarr::structure_with_blob<T>`
    - resizing methods are removed
    - same as above:
        - `length`, `blob_size`
        - `fix`, `fixs`, `fixt`
    - `get()` get the fixed item inside the attached blob
- `noarr::blob`
    - implementation can be swapped in `noarr::blob::driver` (how to alloc, how to copy)
    - remembered data
        - `void* data` data pointer, can be null
        - `bit onDevice` device or host?    (bits are combined into a flags byte)
        - `bit ownsData` free the data during destruction?
    - `::from_host_pointer(void*)` creates a reference blob from a host pointer
    - `::from_device_pointer(void*)` creates a reference blob from a device pointer
    - `::device_allocate(std::size_t)` creates new device blob
    - `::host_allocate(std::size_t)` creates new host blob
    - `.fill_with(blob | void*)` copy data from somewhere to me
    - `.copy_to(blob | void*)` copy data from me to somewhere
    - `.free()` frees data, sets poitner to null, called from the destructor

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

template<typename TChunkHistograms, typename TFinalHistogram>
__global__ void reduce_histograms(
    TChunkHistograms chunkHistograms
    TFinalHistogram finalHistogram
) {
    int histogramBin = threadIdx.x;
    int histogramCount = chunkHistograms.length<'c'>();

    // sum one bin over all chunks
    int sum = 0;
    for (int c = 0; c < histogramCount; c++) {
        sum += chunkHistograms.fixs<'c', 'b'>(c, histogramBin).get();
    }

    finalHistogram.fix<'b'>(histogramBin).get() = sum;
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

    // re-interpret the given blob with noarr as a sequence of chunks
    // c = chunk
    // p = pixel
    auto s_pixels = noarr::structure<
        noarr::vector<'c', noarr::vector<'p', noarr::scalar<char>>>
    >()
        .set_length<'c'>(chunkSize)
        .set_length<'p'>(chunkSize);

    // create structure for chunk histograms
    // c = chunk
    // b = bin
    auto s_chunkHistograms = noarr::structure<
        noarr::vector<'c', noarr::array<'b', 256, noarr::scalar<int>>>
    >()
        .set_length<'c'>(chunkCount);
    
    // move data onto the device & allocate buffers (pseudocode)
    auto h_pixels = blob::from_host_pointer(pixels);
    auto d_pixels = blob::device_allocate(width * height).fill_with(pixels);
    //h_pixels.copy_to(d_pixels); // also could be used for data movement

    auto h_chunkHistograms = blob::host_allocate(s_chunkHistograms.blob_size());
    auto d_chunkHistograms = blob::device_allocate(s_chunkHistograms.blob_size());

    auto s_finalHistogram = noarr::structure<
        noarr::array<'b', 256, noarr::scalar<int>>
    >();
    auto d_finalHistogram = blob::device_allocate(s_finalHistogram.blob_size());

    // run kernels
    reduce_chunks<<<1, chunkCount>>>(
        s_pixels.with_blob(d_pixels),
        s_chunkHistograms.with_blob(d_chunkHistograms)
    );
    reduce_histograms<<<1, 256>>>(
        s_chunkHistograms.with_blob(d_chunkHistograms),
        s_finalHistogram.with_blob(d_finalHistogram)
    );

    // get data out of the device
    d_finalHistogram.copy_to(histogram);

    // buffers are freed up during destruction
    // (only those that were allocated, not referenced)
}
```
