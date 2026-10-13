#include "lauxlib.h"

static int f_gc (lua_State *L);
static int f_tostring (lua_State *L);

static const luaL_Reg metameth[] = {
  {"__index", NULL},
  {"__gc", f_gc},
  {"__close", f_gc},
  {"__tostring", f_tostring},
  {NULL, NULL}
};
