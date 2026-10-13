#include "lstate.h"
#include "cclosure.h"
#include "fn_luaC_newobj.h"

CClosure *luaF_newCclosure (lua_State *L, int nupvals) {
  GCObject *o = luaC_newobj(L, 38, (int)sizeof(CClosure) + (int)sizeof(TValue) * (nupvals - 1));
  CClosure *c = (CClosure *)o;
  c->nupvalues = (lu_byte)nupvals;
  return c;
}
