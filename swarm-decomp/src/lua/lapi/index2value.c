#include "lstate.h"
#include "cclosure.h"
#include "lua.h"

#define G(L) ((L)->l_G)

static TValue *index2value (lua_State *L, int idx) {
  CallInfo *ci = L->ci;
  if (idx > 0) {
    StkId o = ci->func + idx;
    if (o >= L->top) return &G(L)->nilvalue;
    else return o;
  }
  else if (idx > LUA_REGISTRYINDEX) {
    return L->top + idx;
  }
  else if (idx == LUA_REGISTRYINDEX)
    return &G(L)->l_registry;
  else {
    idx = LUA_REGISTRYINDEX - idx;
    if (ci->func->tt_ == (6 | (2 << 4) | (1 << 6))) {
      CClosure *func = (CClosure *)ci->func->value_.gc;
      return (idx <= func->nupvalues) ? &func->upvalue[idx-1] : &G(L)->nilvalue;
    }
    else
      return &G(L)->nilvalue;
  }
}
