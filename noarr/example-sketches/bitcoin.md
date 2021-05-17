# Bitcoin example sketch


**Demonstrates:**

- parallel execution on multiple devices
- parallel read from one buffer, distributed to all devices


**Description:**

The input is a binary message. It is loaded into memory on each device and kernels are executed (one for each device with maximum possible thread count). Each thread tries to generate a random salt, that if appended to the message and hashed would produce a hash with N leading zeros (binary). Basically this is a naive version of the proof of work algorithm of bitcoin.


**Code:**

```cpp
// ...
```
