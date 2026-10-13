#include "lstate.h"
#include "lua.h"

LUA_API void lua_pushvalue (lua_State *L, int idx) {
  StkId top = L->top;
  const TValue *o = index2value(L, idx);
  top->value_ = o->value_;
  top->tt_ = o->tt_;
  L->top = top + 1;
}
