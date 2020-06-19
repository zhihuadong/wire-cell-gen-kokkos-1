# Kokkos version of the wire-cell-gen

## prerequisites
 - Need to have access to a WireCell-Toolkit build.
 - Need to have `Kokkos` installed.

## build

```bash
git clone https://github.com/WireCell/wire-cell-gen-kokkos.git
cd wire-cell-gen-kokkos
./configure /path/to/kokkos /path/to/install
./wcb install
```

## test

run the kokkos unit test manually:

In the source folder (one level above build)
```bash
./build/test_kokkos
```
