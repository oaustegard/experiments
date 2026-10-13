#include "lauxlib.h"
#include "boxgc.h"

static const luaL_Reg boxmt[] = {
  {"__gc", boxgc},
  {"__close", boxgc},
  {NULL, NULL}
};
