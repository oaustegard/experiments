#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushboolean (lua_State *L, int b) {
  L->top->tt_ = b ? 0x11 : 0x01;
  L->top++;
}
