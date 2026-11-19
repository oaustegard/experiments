set -e
cd /tmp/jail/tess534
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TRAINING_TOOLS=OFF -DGRAPHICS_DISABLED=ON \
  -DDISABLE_ARCHIVE=ON -DDISABLE_CURL=ON -DOPENMP_BUILD=OFF -DBUILD_SHARED_LIBS=OFF -DSW_BUILD=OFF \
  -DENABLE_NATIVE=OFF -DBUILD_TESTS=OFF > build_cmake.log 2>&1
cmake --build build -j4 --target libtesseract > build_make.log 2>&1
I=""
for d in src/lstm src/ccutil src/ccstruct src/arch src/ccmain src/dict src/viewer src/classify src/textord src/cutil src/wordrec src/training/common include build/include build; do I="$I -I$d"; done
g++ -O2 -std=c++17 -DGRAPHICS_DISABLED -DFAST_FLOAT $I $(pkg-config --cflags lept) harness/tess_dump.cpp \
  $(find build -name 'libtesseract*.a' | head -1) $(pkg-config --libs lept) -lpthread -o harness/tess_dump > build_harness.log 2>&1
echo built
