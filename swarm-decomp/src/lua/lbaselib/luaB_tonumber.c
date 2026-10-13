#include "lua_api.h"

static int luaB_tonumber (lua_State *L) {
  if (lua_type(L, 2) <= 0) {
    if (lua_type(L, 1) == LUA_TNUMBER) {
      lua_settop(L, 1);
      return 1;
    }
    else {
      size_t l;
      const char *s = lua_tolstring(L, 1, &l);
      if (s != NULL && lua_stringtonumber(L, s) == l + 1)
        return 1;
      luaL_checkany(L, 1);
    }
  }
  else {
    size_t l;
    const char *s;
    lua_Integer n = 0;
    lua_Integer base = luaL_checkinteger(L, 2);
    luaL_checktype(L, 1, LUA_TSTRING);
    s = lua_tolstring(L, 1, &l);
    luaL_argcheck(L, 2 <= base && base <= 36, 2, "base out of range");
    if (b_str2int(s, (int)base, &n) == s + l) {
      lua_pushinteger(L, n);
      return 1;
    }
  }
  lua_pushnil(L);
  return 1;
}
