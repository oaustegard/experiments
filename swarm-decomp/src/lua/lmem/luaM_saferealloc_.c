#include "fn_luaD_throw.h"
#include "fn_luaM_realloc_.h"
#include "fn_luaM_saferealloc_.h"

void *luaM_saferealloc_ (lua_State *L, void *block, size_t osize, size_t nsize) {
  void *newblock = luaM_realloc_(L, block, osize, nsize);
  if (__builtin_expect(newblock == NULL && nsize > 0, 0))
    luaD_throw(L, LUA_ERRMEM);
  return newblock;
}
