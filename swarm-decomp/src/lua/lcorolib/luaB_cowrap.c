#include "lua_api.h"
#include "cosstatus.h"

static int luaB_cocreate (lua_State *L);
static int luaB_auxwrap (lua_State *L);

static int luaB_cowrap (lua_State *L) {
  luaB_cocreate(L);
  lua_pushcclosure(L, luaB_auxwrap, 1);
  return 1;
}
