#include "lua_api.h"

static int luaB_collectgarbage (lua_State *L) {
  static const char *const opts[] = {"stop", "restart", "collect",
    "count", "step", "setpause", "setstepmul",
    "isrunning", "generational", "incremental", NULL};
  static const int optsnum[] = {LUA_GCSTOP, LUA_GCRESTART, LUA_GCCOLLECT,
    LUA_GCCOUNT, LUA_GCSTEP, LUA_GCSETPAUSE, LUA_GCSETSTEPMUL,
    LUA_GCISRUNNING, LUA_GCGEN, LUA_GCINC};
  int o = optsnum[luaL_checkoption(L, 1, "collect", opts)];
  switch (o) {
    case LUA_GCCOUNT: {
      int k = lua_gc(L, o);
      int b = lua_gc(L, LUA_GCCOUNTB);
      if (k == -1) break;
      lua_pushnumber(L, (lua_Number)k + ((lua_Number)b / 1024));
      return 1;
    }
    case LUA_GCSTEP: {
      int step = (int)luaL_optinteger(L, 2, 0);
      int res = lua_gc(L, o, step);
      if (res == -1) break;
      lua_pushboolean(L, res);
      return 1;
    }
    case LUA_GCSETPAUSE:
    case LUA_GCSETSTEPMUL: {
      int p = (int)luaL_optinteger(L, 2, 0);
      int res = lua_gc(L, o, p);
      if (res == -1) break;
      lua_pushinteger(L, res);
      return 1;
    }
    case LUA_GCISRUNNING: {
      int res = lua_gc(L, o);
      if (res == -1) break;
      lua_pushboolean(L, res);
      return 1;
    }
    case LUA_GCGEN: {
      int minormul = (int)luaL_optinteger(L, 2, 0);
      int majormul = (int)luaL_optinteger(L, 3, 0);
      int res = lua_gc(L, LUA_GCGEN, minormul, majormul);
      if (res == -1) break;
      lua_pushstring(L, (res == LUA_GCINC) ? "incremental" : "generational");
      return 1;
    }
    case LUA_GCINC: {
      int pause = (int)luaL_optinteger(L, 2, 0);
      int stepmul = (int)luaL_optinteger(L, 3, 0);
      int stepsize = (int)luaL_optinteger(L, 4, 0);
      int res = lua_gc(L, LUA_GCINC, pause, stepmul, stepsize);
      if (res == -1) break;
      lua_pushstring(L, (res == LUA_GCINC) ? "incremental" : "generational");
      return 1;
    }
    default: {
      int res = lua_gc(L, o);
      if (res == -1) break;
      lua_pushinteger(L, res);
      return 1;
    }
  }
  lua_pushnil(L);
  return 1;
}
