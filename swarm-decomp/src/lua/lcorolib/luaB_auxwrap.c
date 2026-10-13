#include "lua_api.h"
#include "cosstatus.h"
#include "auxresume.h"

static int luaB_auxwrap (lua_State *L) {
  lua_State *co = lua_tothread(L, lua_upvalueindex(1));
  int r = auxresume(L, co, lua_gettop(L));
  if (r < 0) {
    int stat = lua_status(co);
    if (stat != LUA_OK && stat != LUA_YIELD) {
      stat = lua_closethread(co, L);
      lua_xmove(co, L, 1);
    }
    if (stat != LUA_ERRMEM && lua_type(L, -1) == LUA_TSTRING) {
      luaL_where(L, 1);
      lua_insert(L, -2);
      lua_concat(L, 2);
    }
    return lua_error(L);
  }
  return r;
}
