#include "lstate.h"
#include "fn_luaM_free_.h"

void luaE_shrinkCI (lua_State *L) {
  CallInfo *ci = L->ci->next;
  CallInfo *next;
  if (ci == NULL)
    return;
  while ((next = ci->next) != NULL) {
    CallInfo *next2 = next->next;
    ci->next = next2;
    L->nci--;
    luaM_free_(L, next, sizeof(CallInfo));
    if (next2 == NULL)
      break;
    else {
      next2->previous = ci;
      ci = next2;
    }
  }
}
