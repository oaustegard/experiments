#include "lstate.h"
#include "fn_luaH_get.h"
#include "lua.h"

LUA_API int lua_rawgetp (lua_State *L, int idx, const void *p) {
  Table *t;
  TValue k;
  const TValue *val;
  Value v;
  lu_byte tt;
  t = (Table *)index2value(L, idx)->value_.gc;
  k.value_.p = (void *)p;
  k.tt_ = 2;
  val = luaH_get(t, &k);
  tt = val->tt_;
  v = val->value_;
  if ((tt & 0xf) == 0) {
    L->top->tt_ = 0;
    L->top++;
    return 0;
  }
  else {
    L->top->value_ = v;
    L->top->tt_ = tt;
    L->top++;
    return tt & 0xf;
  }
}
