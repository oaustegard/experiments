#include <string.h>
#include <lua.h>
#include <lauxlib.h>

static lua_State *getthread (lua_State *L, int *arg);
static void checkstack (lua_State *L, lua_State *L1, int n);
static void hookf (lua_State *L, lua_Debug *ar);

#define HOOKKEY "_HOOKKEY"

static int db_sethook (lua_State *L) {
  int arg, mask, count;
  lua_Hook func;
  lua_State *L1 = getthread(L, &arg);
  if (lua_isnoneornil(L, arg+1)) {
    lua_settop(L, arg+1);
    func = NULL; mask = 0; count = 0;
  }
  else {
    const char *smask = luaL_checklstring(L, arg+2, NULL);
    luaL_checktype(L, arg+1, LUA_TFUNCTION);
    count = (int)luaL_optinteger(L, arg + 3, 0);
    func = hookf;
    mask = 0;
    if (strchr(smask, 'c')) mask |= LUA_MASKCALL;
    if (strchr(smask, 'r')) mask |= LUA_MASKRET;
    if (strchr(smask, 'l')) mask |= LUA_MASKLINE;
    if (count > 0) mask |= LUA_MASKCOUNT;
  }
  if (!luaL_getsubtable(L, LUA_REGISTRYINDEX, HOOKKEY)) {
    lua_pushstring(L, "k");
    lua_setfield(L, -2, "__mode");
    lua_pushvalue(L, -1);
    lua_setmetatable(L, -2);
  }
  checkstack(L, L1, 1);
  lua_pushthread(L1); lua_xmove(L1, L, 1);
  lua_pushvalue(L, arg+1);
  lua_rawset(L, -3);
  lua_sethook(L1, func, mask, count);
  return 0;
}
