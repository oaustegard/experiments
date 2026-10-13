#include "fn_luaD_throw.h"
#include "fn_luaG_runerror.h"
#include "fn_luaM_realloc_.h"
#include "fn_luaM_growaux_.h"

void *luaM_growaux_ (lua_State *L, void *block, int nelems, int *psize,
                     int size_elem, int limit, const char *what) {
  void *newblock;
  int size = *psize;
  int newsize = limit;
  size_t nsize;
  if (size > nelems)
    return block;
  if (size >= limit / 2) {
    if (size >= limit)
      luaG_runerror(L, "too many %s (limit is %d)", what, limit);
  }
  else {
    newsize = size * 2;
    if (newsize < 4)
      newsize = 4;
  }
  nsize = (size_t)newsize * size_elem;
  newblock = luaM_realloc_(L, block, (size_t)size * size_elem, nsize);
  if (__builtin_expect(newblock == NULL && nsize > 0, 0))
    luaD_throw(L, LUA_ERRMEM);
  *psize = newsize;
  return newblock;
}
