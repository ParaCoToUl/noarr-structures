# Sobel example sketch


**Demonstrates:**

- parallel execution on one device
- parallel read from one buffer
- double-buffering


**Description:**

The input is a sequence of N images of various resolutions with a reasonable upper limit on the resolution (to allocate buffers). For each of these images a sobel operator is calculated and saved to the output as two images (vertical edges & horizontal edges). The two edge-detection kernels run in parallel, while reading from the same source image. All images are grayscale for simplicity. This algorithm is not exactly Sobel, but it's close enough.


**Code:**

```cpp
// ...
```
