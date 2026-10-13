#include "lstate.h"
#include "lua.h"
#include "StackValue.h"

LUA_API int lua_getiuservalue (lua_State *L, int idx, int n) {
  TValue *o = index2value(L, idx);
  int t;
  if (n <= 0 || n > ((Udata *)o->value_.gc)->nuvalue) {
    setnilvalue(&((StackValue *)L->top)->val);
    t = LUA_TNONE;
  }
  else {
    const TValue *uv = &((Udata *)o->value_.gc)->uv[n - 1].uv;
    TValue *tv = &((StackValue *)L->top)->val;
    tv->value_ = uv->value_;
    tv->tt_ = uv->tt_;
    t = tv->tt_ & 0xf;
  }
  L->top++;
  return t;
}
