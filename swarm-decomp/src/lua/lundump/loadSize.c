#include "lundump.h"

static size_t loadSize (LoadState *S) {
  return loadUnsigned(S, (size_t)-1);
}
