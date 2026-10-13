#include "lua_api.h"

static int auxresume (lua_State *L, lua_State *co, int narg) {
  int status, nres;
  if (luai_unlikely(!lua_checkstack(co, narg))) {
    lua_pushstring(L, "too many arguments to resume");
    return -1;
  }
  lua_xmove(L, co, narg);
  status = lua_resume(co, L, narg, &nres);
  if (luai_likely(status == LUA_OK || status == LUA_YIELD)) {
    if (luai_unlikely(!lua_checkstack(L, nres + 1))) {
      lua_settop(co, -nres - 1);
      lua_pushstring(L, "too many results to resume");
      return -1;
    }
    lua_xmove(co, L, nres);
    return nres;
  }
  else {
    lua_xmove(co, L, 1);
    return -1;
  }
}
