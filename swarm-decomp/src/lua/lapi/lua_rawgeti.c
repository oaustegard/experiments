#include "lstate.h"
#include "fn_luaH_getint.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API int lua_rawgeti (lua_State *L, int idx, lua_Integer n) {
  Table *t;
  const TValue *slot;
  Value v;
  lu_byte tt;
  t = (Table *)index2value(L, idx)->value_.gc;
  slot = luaH_getint(t, n);
  v = slot->value_;
  tt = slot->tt_;
  if ((tt & 0xf) == 0) {
    L->top->tt_ = 0;
  }
  else {
    L->top->value_ = v;
    L->top->tt_ = tt;
  }
  L->top++;
  return (L->top[-1].tt_ & 0xf);
}
