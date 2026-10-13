#include "lua_api.h"
#include "cosstatus.h"
#include "getco.h"
#include "auxstatus.h"



static int luaB_close (lua_State *L) {
  lua_State *co = getco(L);
  int status = auxstatus(L, co);
  switch (status) {
    case COS_DEAD: case COS_YIELD: {
      status = lua_closethread(co, L);
      if (status == LUA_OK) {
        lua_pushboolean(L, 1);
        return 1;
      }
      else {
        lua_pushboolean(L, 0);
        lua_xmove(co, L, 1);
        return 2;
      }
    }
    default: {
      return luaL_error(L, "cannot close a %s coroutine", statname[status]);
    }
  }
}
