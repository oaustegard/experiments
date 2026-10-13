#include "lauxlib.h"

LUALIB_API void luaL_setfuncs (lua_State *L, const luaL_Reg *l, int nup) {
  if (!lua_checkstack(L, nup))
    luaL_error(L, "stack overflow (%s)", "too many upvalues");
  for (; l->name != NULL; l++) {
    if (l->func == NULL)
      lua_pushboolean(L, 0);
    else {
      int i;
      for (i = 0; i < nup; i++)
        lua_pushvalue(L, -nup);
      lua_pushcclosure(L, l->func, nup);
    }
    lua_setfield(L, -(nup + 2), l->name);
  }
  lua_pop(L, nup);
}
