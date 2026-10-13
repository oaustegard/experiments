#include <stdint.h>
#include "lauxlib.h"
#include "newbuffsize.h"

static size_t newbuffsize (luaL_Buffer *B, size_t sz) {
  size_t newsize = (B->size / 2) * 3;
  if (SIZE_MAX - sz < B->n)
    return luaL_error(B->L, "buffer too large");
  if (newsize < B->n + sz)
    newsize = B->n + sz;
  return newsize;
}
