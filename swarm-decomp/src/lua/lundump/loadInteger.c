#include "lundump.h"

static lua_Integer loadInteger (LoadState *S) {
  lua_Integer x;
  loadBlock(S, &x, sizeof(x));
  return x;
}
