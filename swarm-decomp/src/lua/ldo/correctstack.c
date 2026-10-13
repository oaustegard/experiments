#include "lstate.h"
#include "UpVal.h"
#include "StkIdRel.h"
#include "correctstack.h"

#define isLua(ci) (!((ci)->callstatus & 2))

static void correctstack (lua_State *L) {
  UpVal *up;
  CallInfo *ci;
  L->top = (StkId)((char *)L->stack + (size_t)L->top);
  L->tbclist = (StkId)((char *)L->stack + (size_t)L->tbclist);
  for (up = L->openupval; up != NULL; up = up->u.open.next)
    up->v.p = (TValue *)((char *)L->stack + (size_t)up->v.p);
  for (ci = L->ci; ci != NULL; ci = ci->previous) {
    ci->top = (StkId)((char *)L->stack + ((StkIdRel *)&ci->top)->offset);
    ci->func = (StkId)((char *)L->stack + ((StkIdRel *)&ci->func)->offset);
    if (isLua(ci))
      ci->u.l.trap = 1;
  }
}
