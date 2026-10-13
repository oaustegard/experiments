#include "lauxlib.h"

void luaL_pushresultsize (luaL_Buffer *B, size_t sz) {
  B->n += sz;
  luaL_pushresult(B);
}
