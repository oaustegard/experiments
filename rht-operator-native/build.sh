#!/usr/bin/env bash
# Build the kernel variants. -ffp-contract=off is required for the bit-exactness
# claim: it forbids fusing a multiply into an add.
set -euo pipefail
cd "$(dirname "$0")"
CC=${CC:-cc}
FL="-O3 -ffp-contract=off -fPIC -shared"
OMP="-fopenmp"
echo 'int main(){return 0;}' > /tmp/_omp.c
$CC -fopenmp /tmp/_omp.c -o /tmp/_omp 2>/dev/null || OMP=""
ARCH="-march=native"
$CC $ARCH /tmp/_omp.c -o /tmp/_omp 2>/dev/null || ARCH="-mcpu=native"
$CC $ARCH /tmp/_omp.c -o /tmp/_omp 2>/dev/null || ARCH=""
$CC $FL $ARCH $OMP -o rht_kernel_native.so rht_kernel.c
$CC $FL $OMP -o rht_kernel_sse2.so rht_kernel.c      # compiler-default ISA ("portable")
$CC -O0 -fPIC -shared -o rht_kernel_O0.so rht_kernel.c
$CC $FL $ARCH -o rht_kernel_nativenc.so rht_kernel.c
echo "built: arch='${ARCH}' omp='${OMP}'"
