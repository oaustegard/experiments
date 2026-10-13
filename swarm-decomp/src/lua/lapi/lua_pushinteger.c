#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushinteger (lua_State *L, lua_Integer n) {
  L->top->value_.i = n;
  L->top->tt_ = 3;
  L->top++;
}
