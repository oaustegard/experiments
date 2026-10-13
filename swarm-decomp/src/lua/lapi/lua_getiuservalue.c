#include "lstate.h"
#include "lua.h"

LUA_API int lua_getiuservalue (lua_State *L, int idx, int n) {
  TValue *o = index2value(L, idx);
  int t;
  if (n <= 0 || n > ((Udata *)o->value_.gc)->nuvalue) {
    setnilvalue(L->top);
    t = LUA_TNONE;
  }
  else {
    const TValue *uv = &((Udata *)o->value_.gc)->uv[n - 1].uv;
    L->top->value_ = uv->value_;
    L->top->tt_ = uv->tt_;
    t = L->top->tt_ & 0xf;
  }
  L->top++;
  return t;
}
