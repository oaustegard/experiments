set -e
cd /tmp/jail/tess534
I=""
for d in src/lstm src/ccutil src/ccstruct src/arch src/ccmain src/dict src/viewer src/classify src/textord src/cutil src/wordrec include build/include build; do I="$I -I$d"; done
g++ -O2 -std=c++17 -DGRAPHICS_DISABLED -DFAST_FLOAT $I $(pkg-config --cflags lept) harness/tess_layers.cpp \
  $(find build -name 'libtesseract*.a' | head -1) $(pkg-config --libs lept) -lpthread -o harness/tess_layers > build_layers.log 2>&1
echo built
