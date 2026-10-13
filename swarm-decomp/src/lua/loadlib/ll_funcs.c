#include "lua.h"
#include "lauxlib.h"
#include "ll_require.h"

static const luaL_Reg ll_funcs[] = {
  {"require", ll_require},
  {NULL, NULL}
};
