#include "lstate.h"
#include "lua.h"

static StkId index2stack (lua_State *L, int idx) {
  CallInfo *ci = L->ci;
  if (idx > 0) {
    StkId o = ci->func + idx;
    return o;
  }
  else return L->top + idx;
}
