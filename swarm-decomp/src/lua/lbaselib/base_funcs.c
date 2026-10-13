#include "lua_api.h"

static int luaB_assert (lua_State *L);
static int luaB_collectgarbage (lua_State *L);
static int luaB_dofile (lua_State *L);
static int luaB_error (lua_State *L);
static int luaB_getmetatable (lua_State *L);
static int luaB_ipairs (lua_State *L);
static int luaB_loadfile (lua_State *L);
static int luaB_load (lua_State *L);
static int luaB_next (lua_State *L);
static int luaB_pairs (lua_State *L);
static int luaB_pcall (lua_State *L);
static int luaB_print (lua_State *L);
static int luaB_warn (lua_State *L);
static int luaB_rawequal (lua_State *L);
static int luaB_rawlen (lua_State *L);
static int luaB_rawget (lua_State *L);
static int luaB_rawset (lua_State *L);
static int luaB_select (lua_State *L);
static int luaB_setmetatable (lua_State *L);
static int luaB_tonumber (lua_State *L);
static int luaB_tostring (lua_State *L);
static int luaB_type (lua_State *L);
static int luaB_xpcall (lua_State *L);

static const luaL_Reg base_funcs[] = {
  {"assert", luaB_assert},
  {"collectgarbage", luaB_collectgarbage},
  {"dofile", luaB_dofile},
  {"error", luaB_error},
  {"getmetatable", luaB_getmetatable},
  {"ipairs", luaB_ipairs},
  {"loadfile", luaB_loadfile},
  {"load", luaB_load},
  {"next", luaB_next},
  {"pairs", luaB_pairs},
  {"pcall", luaB_pcall},
  {"print", luaB_print},
  {"warn", luaB_warn},
  {"rawequal", luaB_rawequal},
  {"rawlen", luaB_rawlen},
  {"rawget", luaB_rawget},
  {"rawset", luaB_rawset},
  {"select", luaB_select},
  {"setmetatable", luaB_setmetatable},
  {"tonumber", luaB_tonumber},
  {"tostring", luaB_tostring},
  {"type", luaB_type},
  {"xpcall", luaB_xpcall},
  {"_G", NULL},
  {"_VERSION", NULL},
  {NULL, NULL}
};
