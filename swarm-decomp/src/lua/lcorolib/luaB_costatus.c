#include "lua_api.h"
#include "cosstatus.h"
#include "getco.h"
#include "auxstatus.h"



static int luaB_costatus (lua_State *L) {
  lua_State *co = lua_tothread(L, 1);
  luaL_argexpected(L, co, 1, "thread");
  lua_pushstring(L, statname[auxstatus(L, co)]);
  return 1;
}
