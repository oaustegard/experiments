#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushnil (lua_State *L) {
  L->top->tt_ = 0;
  L->top++;
}
