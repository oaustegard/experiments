#include "lstate.h"
#include "fn_luaC_barrierback_.h"
#include "fn_luaH_get.h"
#include "fn_luaV_finishset.h"
#include "lua.h"

LUA_API void lua_settable (lua_State *L, int idx) {
  TValue *t;
  const TValue *slot = NULL;
  t = index2value(L, idx);
  if (t->tt_ == 0x45 &&
      (slot = luaH_get((Table *)t->value_.gc, L->top - 2), (slot->tt_ & 0xf) != 0)) {
    TValue *s = (TValue *)slot;
    TValue *v = L->top - 1;
    s->value_ = v->value_;
    s->tt_ = v->tt_;
    if ((v->tt_ & 0x40) && (t->value_.gc->marked & 0x20) && (v->value_.gc->marked & 0x18))
      luaC_barrierback_(L, t->value_.gc);
  }
  else
    luaV_finishset(L, t, L->top - 2, L->top - 1, slot);
  L->top -= 2;
}
