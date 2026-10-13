#include "lua_api.h"

static int luaB_yield (lua_State *L) {
  return lua_yield(L, lua_gettop(L));
}
