#include "lstate.h"
#include "LClosure.h"
#include "fn_luaC_newobj.h"

LClosure *luaF_newLclosure (lua_State *L, int nupvals) {
  GCObject *o = luaC_newobj(L, 6, (int)sizeof(LClosure) + (int)sizeof(struct UpVal *) * (nupvals - 1));
  LClosure *c = (LClosure *)o;
  c->p = NULL;
  c->nupvalues = (lu_byte)nupvals;
  while (nupvals--)
    c->upvals[nupvals] = NULL;
  return c;
}
