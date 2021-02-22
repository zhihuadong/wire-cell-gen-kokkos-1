# Kokkos version of the wire-cell-gen

## prerequisites
 - Need to have access to a `Wire-Cell Toolkit` build and its dependencies.
 - Need to have access to a `Kokkos` build.

## build on cori with shifter

[Shifter usage for Cori](https://github.com/hep-cce2/PPSwork/blob/master/Wire-Cell/Shifter.md)

Within shifter container:

```bash
git clone https://github.com/WireCell/wire-cell-gen-kokkos.git
cd wire-cell-gen-kokkos
./configure /path/to/kokkos /path/to/install
./wcb install
```

**To build on other machines, change the `configure` to find the needed dependencies.**

example configuration for using Kokkos-cuda on Cori:
```
KOKKOS_PATH=$1
INSTALL=$2

KOKKOS_INC=$KOKKOS_PATH/include
KOKKOS_LIB=$KOKKOS_PATH/lib64

./wcb configure --prefix=$INSTALL \
--with-tbb="$TBBROOT" \
--with-jsoncpp="$JSONCPP_FQ_DIR" \
--with-jsonnet="$JSONNET_FQ_DIR" \
--with-eigen-include="<path-to-eigen-3.3.7>" \
--with-root="$ROOTSYS" \
--with-fftw="$FFTW_FQ_DIR" \
--with-fftw-include="$FFTW_INC" \
--with-fftw-lib="$FFTW_LIBRARY" \
--with-fftwthreads="$FFTW_FQ_DIR" \
--boost-includes="$BOOST_INC" \
--boost-libs="$BOOST_LIB" \
--boost-mt \
--with-cuda="$CUDA_PATH/" \
--with-cuda-lib="$CUDA_PATH/lib64" \
--with-wct=$WIRECELL_FQ_DIR/ \
--with-wct-lib=$WIRECELL_LIB \
--with-kokkos=$KOKKOS_PATH/ \
--with-kokkos-include=$KOKKOS_INC/ \
--with-kokkos-lib=$KOKKOS_LIB/ \
--kokkos-options="cuda" \
--with-spdlog-lib=$SPDLOG_LIB \
--with-spdlog-include=$SPDLOG_INC \
```

**note: Eigen 3.3.7 is needed to compile source code including Eigen with nvcc**

## unit test

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

## full test running `wire-cell` as plugin of `LArSoft`


### download `wire-cell-data`

`wire-cell-data` contains needed data files, e.g. geometry files, running full tests.

```
git clone https://github.com/WireCell/wire-cell-data.git
```

### $WIRECELL_PATH

`wire-cell` searches pathes in this env var for configuration and data files.

for bash, run something like this below:

```
export WIRECELL_PATH=$WIRECELL_FQ_DIR/wirecell-0.14.0/cfg # main cfg
export WIRECELL_PATH=$WIRECELL_DATA_PATH:$WIRECELL_PATH # data
export WIRECELL_PATH=$WIRECELL_GEN_KOKKOS_INSTALL_PATH/share/wirecell/cfg:$WIRECELL_PATH # gen-kokkos

```
Variable meaning:
 - `$WIRECELL_FQ_DIR` is a variable defined developing in Kyle's container or `setup wirecell` in a Fermilab ups system, current version is `0.14.0`, may upgrade in the future.
 - `WIRECELL_DATA_PATH` refer to the git repository cloned from the previous step
 - `WIRECELL_GEN_KOKKOS_INSTALL_PATH` refer to the install path of the `wire-cell-gen-kokkos` standalone package.

### $LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=$WIRECELL_GEN_KOKKOS_INSTALL_PATH/lib64/:$LD_LIBRARY_PATH
```


### run

 - input: a root file (refered to as [`g4.root`](https://github.com/hep-cce2/PPSwork/blob/master/Wire-Cell/examples/g4.root) below) containing Geant4 energy depo (`sim::SimEnergyDeposits`)
 - in the example folder: `lar -n 1 -c sim.fcl g4.root`

## test origin CPU/CUDA implementation

 - download [this branch](https://github.com/HaiwangYu/wire-cell-toolkit/tree/debug_cuda_gen) of `wire-cell-toolkit`
   - the source code location is referred as `<wire-cell-toolkit-cuda-source>`
 - use the [`configure`](https://github.com/HaiwangYu/wire-cell-toolkit/blob/debug_cuda_gen/configure) to configue
   - specify install location for this script, referted as `<wire-cell-toolkit-cuda-install>` later
   - default use CUDA
   - remove [these two lines](https://github.com/HaiwangYu/wire-cell-toolkit/blob/debug_cuda_gen/configure#L22-L23) to use CPU
 - use this cmd to compile and install `./wcb -p --notests install`
 - modify `LD_LIBRARY_PATH` and `WIRECELL_PATH` to use the library and jsonnet configuration
```
export LD_LIBRARY_PATH=<wire-cell-toolkit-cuda-install>/lib64:$LD_LIBRARY_PATH
export   WIRECELL_PATH=<wire-cell-toolkit-cuda-source>/cfg/:$WIRECELL_PATH
```

 - in the `wire-cell-gen-kokkos/example` folder run this again `lar -n 1 -c sim.fcl g4.root`
