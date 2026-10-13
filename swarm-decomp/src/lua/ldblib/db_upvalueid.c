#include <lua.h>
#include <lauxlib.h>

static void *checkupval (lua_State *L, int argf, int argnup, int *pnup);

static int db_upvalueid (lua_State *L) {
  void *id = checkupval(L, 1, 2, NULL);
  if (id != NULL)
    lua_pushlightuserdata(L, id);
  else
    luaL_pushfail(L);
  return 1;
}
