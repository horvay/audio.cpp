#!/usr/bin/env bash
# Build drama3_batch on ggml's SYCL backend (SYCL + oneMKL over Level Zero, no Vulkan) for the Arc Pro B70.
# Same toolchain as ~/ai/stable-diffusion.cpp-b70-sycl/build_sycl.sh (DPC++ 2025.3, oneMKL 2025.3.1, AOT for BMG G31).
# Run it with D3_BACKEND=sycl (drama3_batch defaults to Vulkan). Usage: ./build_sycl.sh [build-dir]
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BUILD=${1:-$HERE/build/linux-sycl-release}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
MKLROOT=${MKLROOT:-$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
# oneDNN: the same SYCL build the native xpu_sdpa libraries link (libdnnl_h3), so a process has one oneDNN
ONEDNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}
ZE_INC=${ZE_INC:-$HOME/opt/onednn-src-h3/third_party/level_zero}   # level_zero/ze_api.h
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF:$MKLROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# ggml-sycl finds ze_api.h but does not add its directory to the include path
export CPATH=$ZE_INC${CPATH:+:$CPATH}
export MKLROOT CMAKE_PREFIX_PATH=$MKLROOT/lib/cmake/mkl:$CMPLR/lib/cmake/IntelSYCL

# -static-intel: Intel's compiler runtime (libsvml, libimf, libintlc, libirng) linked in; --disable-new-dtags: RPATH, which
# (unlike RUNPATH) also covers the libraries the executable loads; libumf linked directly so the Level Zero UR adapter
# (dlopened by the UR loader, and needing libumf.so.1) finds it already loaded. Without it SYCL sees no GPU.
# ggml-sycl is a static library here, so its own spir64_gen link options never apply: the executable's link has to
# name the B70 target itself, or kernels go missing at run time ("No kernel named ... was found").
cmake -S "$HERE" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DENGINE_ENABLE_VULKAN=OFF -DENGINE_ENABLE_OPENMP=OFF -DGGML_OPENMP=OFF -DGGML_SYCL=ON \
  -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_DEVICE_ARCH=bmg-g31 -DGGML_SYCL_DNN=ON -DDNNL_DIR="$ONEDNN/lib/cmake/dnnl" \
  -DMKL_DIR="$MKLROOT/lib/cmake/mkl" -DMKL_ROOT="$MKLROOT" -DMKL_INTERFACE=ilp64 -DMKL_THREADING=sequential \
  -DLEVEL_ZERO_DEV_INCLUDE_DIR="$ZE_INC" -DZE_LOADER_LIB=/usr/lib/libze_loader.so \
  -DCMAKE_EXE_LINKER_FLAGS="-static-intel -Wl,--disable-new-dtags -L$UMF -Wl,--no-as-needed -lumf -Wl,--as-needed -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen \"-device bmg-g31\"" \
  -DCMAKE_BUILD_RPATH="$CMPLR/lib;$UMF;$MKLROOT/lib;$ONEDNN/lib" -DCMAKE_INSTALL_RPATH="$CMPLR/lib;$UMF;$MKLROOT/lib;$ONEDNN/lib"
# Low priority (nice 10), so it can use most cores: 28 compile jobs, capped at one per 2 GB of free memory (SYCL
# compiles are memory-hungry). JOBS=N overrides.
if [[ -z "${JOBS:-}" ]]; then
  free_gb=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
  JOBS=$(( free_gb / 2 < 28 ? free_gb / 2 : 28 )); (( JOBS >= 3 )) || JOBS=3
fi
echo "building with $JOBS jobs (nice 10)"
nice -n 10 cmake --build "$BUILD" --target drama3_batch audiocpp_cli -j "$JOBS"
ldd "$BUILD/bin/drama3_batch" | grep -Ei 'vulkan|torch|c10' && { echo "unexpected Vulkan/torch dependency" >&2; exit 1; } || true
ldd "$BUILD/bin/drama3_batch" | grep "not found" && { echo "missing shared libraries" >&2; exit 1; } || true
echo "built $BUILD/bin/drama3_batch"
