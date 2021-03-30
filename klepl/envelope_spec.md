# Envelopy

Každá envelope má nějaký pool bufferů (abstrahovaný objektem)

Podporuje následující API:

```C++
envelope.get_buffer(FLAGS);
```

## Flags:

- `env::read`, `env::write`, `env::host`, `env::device(n)`


## Pool

Základní pool se dělí na tři množiny:

- free_buffers
- ready_buffers (pro každý device)
- used_buffers

`get_buffer`, který čte, vyzvedne buffer z množiny ready_buffers, jinak z množiny ready_buffers.

`get_buffer` vrátí handle na fyzický buffer a tento fyzický buffer se uvolní zemřením tohoto handlu a přesune se do správné množiny (po přečtení do free, po zápisu do ready).

zápis bufferu může určit, zda je určen pro jedno čtení (load balancing), či pro čtení z více nodů (to bude doprovozeno tvorbou více virtuálních fyzických bufferů).

## Základní použití

- pool velikosti 1 a zápis pro jedno přečtení
- pool velikosti 2+ a zápis pro jedno přečtení (= double+ buffering či load balancing)
- pool velikosti n a zápis pro (n - 1) přečtení (= třeba meziukládání dat, která někdo jiný počítá)