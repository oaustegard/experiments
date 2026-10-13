#include <lua.h>
#include <lauxlib.h>

static void checkstack (lua_State *L, lua_State *L1, int n) {
  if (luai_unlikely(L != L1 && !lua_checkstack(L1, n)))
    luaL_error(L, "stack overflow");
}
