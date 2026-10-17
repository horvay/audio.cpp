#!/usr/bin/env bash
# ggml's test-backend-ops on the SYCL backend, built standalone from external/ggml with the same toolchain and flags as
# build_sycl.sh: every op is checked against the CPU backend (normalized MSE). Usage after building:
#   build/ggml-tests-sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID      (test mode; takes the B70: run under b70-lease)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD=${1:-$HERE/build/ggml-tests-sycl}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
MKLROOT=${MKLROOT:-$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
# oneDNN: the same SYCL build the native xpu_sdpa libraries link (libdnnl_h3), so a process has one oneDNN
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}
ZE_INC=${ZE_INC:-$HOME/opt/onednn-src-h3/third_party/level_zero}
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF:$MKLROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CPATH=$ZE_INC${CPATH:+:$CPATH}
export MKLROOT CMAKE_PREFIX_PATH=$MKLROOT/lib/cmake/mkl:$CMPLR/lib/cmake/IntelSYCL
# see build_sycl.sh for the linker flags
cmake -S "$HERE/external/ggml" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DBUILD_SHARED_LIBS=OFF \
  -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF -DGGML_OPENMP=OFF -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg-g31 -DGGML_SYCL_DNN=ON -DDNNL_DIR="$ONEDNN/lib/cmake/dnnl" \
  -DMKL_DIR="$MKLROOT/lib/cmake/mkl" -DMKL_ROOT="$MKLROOT" -DMKL_INTERFACE=ilp64 -DMKL_THREADING=sequential \
  -DLEVEL_ZERO_DEV_INCLUDE_DIR="$ZE_INC" -DZE_LOADER_LIB=/usr/lib/libze_loader.so \
  -DCMAKE_EXE_LINKER_FLAGS="-static-intel -Wl,--disable-new-dtags -L$UMF -Wl,--no-as-needed -lumf -Wl,--as-needed -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device bmg-g31\"" \
  -DCMAKE_BUILD_RPATH="$CMPLR/lib;$UMF;$MKLROOT/lib;$ONEDNN/lib"
nice -n 10 cmake --build "$BUILD" --target test-backend-ops -j "${JOBS:-3}"
ldd "$BUILD/bin/test-backend-ops" | grep "not found" && { echo "missing shared libraries" >&2; exit 1; } || true
echo "built $BUILD/bin/test-backend-ops"
