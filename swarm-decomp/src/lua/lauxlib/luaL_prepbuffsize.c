#include "lauxlib.h"
#include "prepbuffsize.h"

char *luaL_prepbuffsize (luaL_Buffer *B, size_t sz) {
  return prepbuffsize(B, sz, -1);
}
