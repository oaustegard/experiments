#include <lua.h>
#include <lauxlib.h>

static lua_State *getthread (lua_State *L, int *arg);
static void checkstack (lua_State *L, lua_State *L1, int n);
static char *unmakemask (int mask, char *smask);
static void hookf (lua_State *L, lua_Debug *ar);

#define HOOKKEY "_HOOKKEY"

static int db_gethook (lua_State *L) {
  int arg;
  lua_State *L1 = getthread(L, &arg);
  char buff[5];
  int mask = lua_gethookmask(L1);
  lua_Hook hook = lua_gethook(L1);
  if (luai_unlikely(hook == NULL)) {
    lua_pushnil(L);
    return 1;
  }
  else if (hook != hookf)
    lua_pushstring(L, "external hook");
  else {
    lua_getfield(L, LUA_REGISTRYINDEX, HOOKKEY);
    checkstack(L, L1, 1);
    lua_pushthread(L1); lua_xmove(L1, L, 1);
    lua_rawget(L, -2);
    lua_remove(L, -2);
  }
  lua_pushstring(L, unmakemask(mask, buff));
  lua_pushinteger(L, lua_gethookcount(L1));
  return 3;
}
