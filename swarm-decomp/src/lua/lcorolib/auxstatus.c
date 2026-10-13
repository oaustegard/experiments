#include "lua_api.h"
#include "cosstatus.h"

static int auxstatus (lua_State *L, lua_State *co) {
  if (L == co) return COS_RUN;
  switch (lua_status(co)) {
    case LUA_YIELD:
      return COS_YIELD;
    case LUA_OK: {
      lua_Debug ar;
      if (lua_getstack(co, 0, &ar)) return COS_NORM;
      else if (lua_gettop(co) == 0) return COS_DEAD;
      else return COS_YIELD;
    }
    default:
      return COS_DEAD;
  }
}
