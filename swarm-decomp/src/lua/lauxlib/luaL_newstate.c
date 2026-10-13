#include <stddef.h>
#include "lua.h"
#include "lauxlib.h"
#include "warnf.h"

lua_State *luaL_newstate(void) {
  lua_State *L = lua_newstate(l_alloc, NULL);
  if (L) {
    lua_atpanic(L, &panic);
    lua_setwarnf(L, warnfoff, L);
  }
  return L;
}
