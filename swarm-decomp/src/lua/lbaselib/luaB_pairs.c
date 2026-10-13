#include "lua_api.h"

static int luaB_next (lua_State *L);
static int pairscont (lua_State *L, int status, lua_KContext k);

static int luaB_pairs (lua_State *L) {
  luaL_checkany(L, 1);
  if (luaL_getmetafield(L, 1, "__pairs") == LUA_TNIL) {
    lua_pushcfunction(L, luaB_next);
    lua_pushvalue(L, 1);
    lua_pushnil(L);
  }
  else {
    lua_pushvalue(L, 1);
    lua_callk(L, 1, 3, 0, pairscont);
  }
  return 3;
}
