#include "lstate.h"
#include "LClosure.h"
#include "UpVal.h"
#include "lgc_defs.h"
#include "fn_luaC_newobj.h"
#include "fn_luaC_barrier_.h"
#include "fn_luaF_initupvals.h"

void luaF_initupvals (lua_State *L, LClosure *cl) {
  int i;
  for (i = 0; i < cl->nupvalues; i++) {
    GCObject *o = luaC_newobj(L, 9, sizeof(UpVal));
    UpVal *uv = (UpVal *)o;
    uv->v.p = &uv->u.value;
    uv->u.value.tt_ = 0;
    cl->upvals[i] = uv;
    if (isblack(cl) && iswhite(o))
      luaC_barrier_(L, (GCObject *)cl, o);
  }
}
