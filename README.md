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
$./build/test_kokkos
  Total size S = 4194304 N = 4096 M = 1024
Kokkos::OpenMP::initialize WARNING: OMP_PROC_BIND environment variable not set
  In general, for best performance with OpenMP 4.0 or better set OMP_PROC_BIND=spread and OMP_PLACES=threads
  For best performance with OpenMP 3.1 set OMP_PROC_BIND=true
  For unit testing set OMP_PROC_BIND=false
  Computed result for 4096 x 1024 is 4194304.000000
  N( 4096 ) M( 1024 ) nrepeat ( 100 ) problem( 33.5954 MB ) time( 0.076382 s ) bandwidth( 43.9834 GB/s )

```
