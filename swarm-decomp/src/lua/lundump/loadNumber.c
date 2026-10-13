#include "lundump.h"

static lua_Number loadNumber (LoadState *S) {
  lua_Number x;
  loadBlock(S, &x, sizeof(x));
  return x;
}
