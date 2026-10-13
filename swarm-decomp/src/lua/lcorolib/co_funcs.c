#include "lua_api.h"
#include "luaB_yield.h"
#include "luaB_corunning.h"
#include "luaB_cocreate.h"
#include "luaB_yieldable.h"
#include "luaB_costatus.h"
#include "luaB_coresume.h"
#include "luaB_close.h"
#include "luaB_cowrap.h"

static const luaL_Reg co_funcs[] = {
  {"create", luaB_cocreate},
  {"resume", luaB_coresume},
  {"running", luaB_corunning},
  {"status", luaB_costatus},
  {"wrap", luaB_cowrap},
  {"yield", luaB_yield},
  {"isyieldable", luaB_yieldable},
  {"close", luaB_close},
  {NULL, NULL}
};
