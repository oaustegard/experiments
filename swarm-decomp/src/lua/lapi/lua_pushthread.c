#include "lstate.h"
#include "lua.h"

LUA_API int lua_pushthread (lua_State *L) {
  L->top->value_.gc = (GCObject *)L;
  L->top->tt_ = 0x48;
  L->top++;
  return (L->l_G->mainthread == L);
}
