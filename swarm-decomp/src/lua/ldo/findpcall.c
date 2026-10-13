#include "lstate.h"

static CallInfo *findpcall (lua_State *L) {
  CallInfo *ci;
  for (ci = L->ci; ci != NULL; ci = ci->previous) {
    if (ci->callstatus & 0x10)
      return ci;
  }
  return NULL;
}
