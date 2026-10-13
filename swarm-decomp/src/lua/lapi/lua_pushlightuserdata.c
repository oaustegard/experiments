#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushlightuserdata (lua_State *L, void *p) {
  L->top->value_.p = p;
  L->top->tt_ = 2;
  L->top++;
}
