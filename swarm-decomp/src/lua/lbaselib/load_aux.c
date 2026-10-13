#include "lua_api.h"

static int load_aux (lua_State *L, int status, int envidx) {
  if (status == LUA_OK) {
    if (envidx != 0) {
      lua_pushvalue(L, envidx);
      if (!lua_setupvalue(L, -2, 1))
        lua_settop(L, -2);
    }
    return 1;
  }
  else {
    lua_pushnil(L);
    lua_rotate(L, -2, 1);
    return 2;
  }
}
