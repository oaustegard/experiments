#include "lstate.h"
#include "lgc_defs.h"
#include "fn_luaC_barrier_.h"
#include "lua.h"

#define isupvalue(i) ((i) < LUA_REGISTRYINDEX)
#define obj2gco(v) ((GCObject *)(v))

LUA_API void lua_copy (lua_State *L, int fromidx, int toidx) {
  TValue *fr, *to;
  fr = index2value(L, fromidx);
  to = index2value(L, toidx);
  setobj(to, fr);
  if (isupvalue(toidx)) {
    LClosure *p = (LClosure *)L->ci->func->value_.gc;
    if (iscollectable(fr) && isblack(p) && iswhite(gcvalue(fr)))
      luaC_barrier_(L, obj2gco(p), gcvalue(fr));
  }
}
