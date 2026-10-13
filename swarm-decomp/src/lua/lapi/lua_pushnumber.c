#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushnumber (lua_State *L, lua_Number n) {
  L->top->value_.n = n;
  L->top->tt_ = 0x13;
  L->top++;
}
