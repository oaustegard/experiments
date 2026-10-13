#include <lua.h>
#include <lauxlib.h>

static int db_debug (lua_State *L);
static int db_getuservalue (lua_State *L);
static int db_gethook (lua_State *L);
static int db_getinfo (lua_State *L);
static int db_getlocal (lua_State *L);
static int db_getregistry (lua_State *L);
static int db_getmetatable (lua_State *L);
static int db_getupvalue (lua_State *L);
static int db_upvaluejoin (lua_State *L);
static int db_upvalueid (lua_State *L);
static int db_setuservalue (lua_State *L);
static int db_sethook (lua_State *L);
static int db_setlocal (lua_State *L);
static int db_setmetatable (lua_State *L);
static int db_setupvalue (lua_State *L);
static int db_traceback (lua_State *L);
static int db_setcstacklimit (lua_State *L);

static const luaL_Reg dblib[] = {
  {"debug", db_debug},
  {"getuservalue", db_getuservalue},
  {"gethook", db_gethook},
  {"getinfo", db_getinfo},
  {"getlocal", db_getlocal},
  {"getregistry", db_getregistry},
  {"getmetatable", db_getmetatable},
  {"getupvalue", db_getupvalue},
  {"upvaluejoin", db_upvaluejoin},
  {"upvalueid", db_upvalueid},
  {"setuservalue", db_setuservalue},
  {"sethook", db_sethook},
  {"setlocal", db_setlocal},
  {"setmetatable", db_setmetatable},
  {"setupvalue", db_setupvalue},
  {"traceback", db_traceback},
  {"setcstacklimit", db_setcstacklimit},
  {NULL, NULL}
};
