#include "lstate.h"
#include "UpVal.h"
#include "lgc_defs.h"
#include "fn_luaC_barrier_.h"
#include "fn_luaF_closeupval.h"

void luaF_closeupval (lua_State *L, StkId level) {
  UpVal *uv;
  StkId upl;
  while ((uv = (UpVal *)L->openupval) != NULL && (upl = uv->v.p) >= level) {
    TValue *slot = &uv->u.value;
    UpVal **prev = uv->u.open.previous;
    UpVal *next = uv->u.open.next;
    *prev = next;
    if (next)
      next->u.open.previous = uv->u.open.previous;
    setobj(slot, uv->v.p);
    uv->v.p = slot;
    if (!iswhite(uv)) {
      uv->marked |= (1 << BLACKBIT);
      if (iscollectable(slot) && iswhite(gcvalue(slot)))
        luaC_barrier_(L, (GCObject *)uv, gcvalue(slot));
    }
  }
}
