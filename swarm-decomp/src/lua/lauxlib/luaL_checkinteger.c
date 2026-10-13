#include "lauxlib.h"
#include "interror.h"

lua_Integer luaL_checkinteger (lua_State *L, int arg) {
  int isnum;
  lua_Integer d = lua_tointegerx(L, arg, &isnum);
  if (__builtin_expect(!isnum, 0))
    interror(L, arg);
  return d;
}
