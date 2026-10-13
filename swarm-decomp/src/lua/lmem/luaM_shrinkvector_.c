#include "fn_luaM_realloc_.h"
#include "fn_luaD_throw.h"

void *luaM_shrinkvector_ (lua_State *L, void *block, int *size, int final_n, int size_elem) {
  void *newblock;
  size_t oldsize = (*size) * size_elem;
  size_t newsize = final_n * size_elem;
  newblock = luaM_realloc_(L, block, oldsize, newsize);
  if (__builtin_expect(newblock == NULL && newsize > 0, 0))
    luaD_throw(L, LUA_ERRMEM);
  *size = final_n;
  return newblock;
}
