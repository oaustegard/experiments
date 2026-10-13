#include "lstate.h"
#include "UpVal.h"
#include "lgc_defs.h"
#include "fn_luaC_newobj.h"
#include "fn_luaC_barrier_.h"

static UpVal *newupval (lua_State *L, StkId level, UpVal **prev) {
  GCObject *o = luaC_newobj(L, 9, sizeof(UpVal));
  UpVal *uv = (UpVal *)o;
  UpVal *next = *prev;
  uv->v.p = level;
  uv->u.open.next = next;
  uv->u.open.previous = prev;
  if (next)
    next->u.open.previous = &uv->u.open.next;
  *prev = uv;
  if (L->twups == L) {
    L->twups = G(L)->twups;
    G(L)->twups = L;
  }
  return uv;
}
