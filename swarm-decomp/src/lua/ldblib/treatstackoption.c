#include <lua.h>
#include <lauxlib.h>

static void treatstackoption (lua_State *L, lua_State *L1, const char *fname) {
  if (L == L1)
    lua_rotate(L, -2, 1);
  else
    lua_xmove(L1, L, 1);
  lua_setfield(L, -2, fname);
}
