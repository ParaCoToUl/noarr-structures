# Autobalance example sketch


**Demonstrates:**

- modifying envelopes in-place and passing them to a different hub
    (= in-place double-buffering)


**Description:**

The input is a sequence of N images of various resolutions with a reasonable upper limit on the resolution (to allocate buffers). Each of these images is send to the GPU, maximum and minimum brightness is computed and then the image is scaled in-place to use the full range of 0-255 brightness levels. The image is then sent back to the CPU. All images are grayscale for simplicity.


**Code:**

```cpp
// ...
```
