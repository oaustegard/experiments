#include "lua_api.h"

static const char *generic_reader (lua_State *L, void *ud, size_t *size) {
  (void)(ud);
  luaL_checkstack(L, 2, "too many nested functions");
  lua_pushvalue(L, 1);
  lua_callk(L, 0, 1, 0, NULL);
  if (lua_type(L, -1) != LUA_TNIL) {
    if (!lua_isstring(L, -1))
      luaL_error(L, "reader function must return a string");
    lua_copy(L, -1, 5);
    lua_settop(L, -2);
    return lua_tolstring(L, 5, size);
  }
  lua_settop(L, -2);
  *size = 0;
  return NULL;
}
