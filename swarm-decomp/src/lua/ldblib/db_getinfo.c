#include <string.h>
#include <lua.h>
#include <lauxlib.h>

static lua_State *getthread (lua_State *L, int *arg);
static void checkstack (lua_State *L, lua_State *L1, int n);
static void treatstackoption (lua_State *L, lua_State *L1, const char *fname);
static void settabss (lua_State *L, const char *k, const char *v);
static void settabsi (lua_State *L, const char *k, int v);
static void settabsb (lua_State *L, const char *k, int v);

static int db_getinfo (lua_State *L) {
  lua_Debug ar;
  int arg;
  lua_State *L1 = getthread(L, &arg);
  const char *options = luaL_optstring(L, arg+2, "flnSrtu");
  checkstack(L, L1, 3);
  luaL_argcheck(L, options[0] != '>', arg + 2, "invalid option '>'");
  if (lua_isfunction(L, arg + 1)) {
    options = lua_pushfstring(L, ">%s", options);
    lua_pushvalue(L, arg + 1);
    lua_xmove(L, L1, 1);
  }
  else {
    if (!lua_getstack(L1, (int)luaL_checkinteger(L, arg + 1), &ar)) {
      lua_pushnil(L);
      return 1;
    }
  }
  if (!lua_getinfo(L1, options, &ar))
    return luaL_argerror(L, arg+2, "invalid option");
  lua_newtable(L);
  if (strchr(options, 'S')) {
    lua_pushlstring(L, ar.source, ar.srclen);
    lua_setfield(L, -2, "source");
    settabss(L, "short_src", ar.short_src);
    settabsi(L, "linedefined", ar.linedefined);
    settabsi(L, "lastlinedefined", ar.lastlinedefined);
    settabss(L, "what", ar.what);
  }
  if (strchr(options, 'l')) {
    settabsi(L, "currentline", ar.currentline);
  }
  if (strchr(options, 'u')) {
    settabsi(L, "nups", ar.nups);
    settabsi(L, "nparams", ar.nparams);
    settabsb(L, "isvararg", ar.isvararg);
  }
  if (strchr(options, 'n')) {
    settabss(L, "name", ar.name);
    settabss(L, "namewhat", ar.namewhat);
  }
  if (strchr(options, 'r')) {
    settabsi(L, "ftransfer", ar.ftransfer);
    settabsi(L, "ntransfer", ar.ntransfer);
  }
  if (strchr(options, 't')) {
    settabsb(L, "istailcall", ar.istailcall);
  }
  if (strchr(options, 'L'))
    treatstackoption(L, L1, "activelines");
  if (strchr(options, 'f'))
    treatstackoption(L, L1, "func");
  return 1;
}
