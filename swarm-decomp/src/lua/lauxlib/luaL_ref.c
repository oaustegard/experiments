#include "lauxlib.h"

#define freelist	(LUA_RIDX_LAST + 1)

LUALIB_API int luaL_ref (lua_State *L, int t) {
  int ref;
  if (lua_isnil(L, -1)) {
    lua_pop(L, 1);
    return LUA_REFNIL;
  }
  t = lua_absindex(L, t);
  if (lua_rawgeti(L, t, freelist) == LUA_TNIL) {
    ref = 0;
    lua_pushinteger(L, 0);
    lua_rawseti(L, t, freelist);
  }
  else
    ref = (int)lua_tointegerx(L, -1, NULL);
  lua_pop(L, 1);
  if (ref != 0) {
    lua_rawgeti(L, t, ref);
    lua_rawseti(L, t, freelist);
  }
  else
    ref = (int)lua_rawlen(L, t) + 1;
  lua_rawseti(L, t, ref);
  return ref;
}
