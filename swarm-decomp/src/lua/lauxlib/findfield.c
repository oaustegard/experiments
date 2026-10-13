#include "lauxlib.h"

static int findfield (lua_State *L, int objidx, int level) {
  if (level == 0 || !lua_istable(L, -1))
    return 0;
  lua_pushnil(L);
  while (lua_next(L, -2)) {
    if (lua_type(L, -2) == LUA_TSTRING) {
      if (lua_rawequal(L, objidx, -1)) {
        lua_pop(L, 1);
        return 1;
      }
      else if (findfield(L, objidx, level - 1)) {
        lua_pushstring(L, ".");
        lua_replace(L, -3);
        lua_concat(L, 3);
        return 1;
      }
    }
    lua_pop(L, 1);
  }
  return 0;
}
