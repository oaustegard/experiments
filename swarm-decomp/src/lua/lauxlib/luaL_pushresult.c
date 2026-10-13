#include "lauxlib.h"

void luaL_pushresult (luaL_Buffer *B) {
  lua_State *L = B->L;
  lua_pushlstring(L, B->b, B->n);
  if (B->b != B->init.b)
    lua_closeslot(L, -2);
  lua_rotate(L, -2, -1);
  lua_settop(L, -2);
}
