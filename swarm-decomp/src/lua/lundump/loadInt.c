#include "lundump.h"

static int loadInt (LoadState *S) {
  return (int)loadUnsigned(S, INT_MAX);
}
