#include <string.h>
#include <lua.h>
#include <lauxlib.h>

static lua_State *getthread (lua_State *L, int *arg);
static void checkstack (lua_State *L, lua_State *L1, int n);
static void treatstackoption (lua_State *L, lua_State *L1, const char *fname);

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
    lua_pushstring(L, ar.short_src);
    lua_setfield(L, -2, "short_src");
    lua_pushinteger(L, ar.linedefined);
    lua_setfield(L, -2, "linedefined");
    lua_pushinteger(L, ar.lastlinedefined);
    lua_setfield(L, -2, "lastlinedefined");
    lua_pushstring(L, ar.what);
    lua_setfield(L, -2, "what");
  }
  if (strchr(options, 'l')) {
    lua_pushinteger(L, ar.currentline);
    lua_setfield(L, -2, "currentline");
  }
  if (strchr(options, 'u')) {
    lua_pushinteger(L, ar.nups);
    lua_setfield(L, -2, "nups");
    lua_pushinteger(L, ar.nparams);
    lua_setfield(L, -2, "nparams");
    lua_pushboolean(L, ar.isvararg);
    lua_setfield(L, -2, "isvararg");
  }
  if (strchr(options, 'n')) {
    lua_pushstring(L, ar.name);
    lua_setfield(L, -2, "name");
    lua_pushstring(L, ar.namewhat);
    lua_setfield(L, -2, "namewhat");
  }
  if (strchr(options, 'r')) {
    lua_pushinteger(L, ar.ftransfer);
    lua_setfield(L, -2, "ftransfer");
    lua_pushinteger(L, ar.ntransfer);
    lua_setfield(L, -2, "ntransfer");
  }
  if (strchr(options, 't')) {
    lua_pushboolean(L, ar.istailcall);
    lua_setfield(L, -2, "istailcall");
  }
  if (strchr(options, 'L')) {
    if (L1 == L)
      lua_rotate(L1, -2, 1);
    else
      lua_xmove(L1, L, 1);
    lua_setfield(L, -2, "activelines");
  }
  if (strchr(options, 'f')) {
    if (L1 == L)
      lua_rotate(L, -2, 1);
    else
      lua_xmove(L1, L, 1);
    lua_setfield(L, -2, "func");
  }
  return 1;
}
