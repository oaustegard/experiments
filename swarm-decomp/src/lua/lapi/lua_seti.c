#include "lstate.h"
#include "fn_luaC_barrierback_.h"
#include "fn_luaH_getint.h"
#include "fn_luaV_finishset.h"
#include "lobject_tags.h"
#include "lua.h"

LUA_API void lua_seti (lua_State *L, int idx, lua_Integer n) {
  TValue *t;
  const TValue *slot = NULL;
  t = index2value(L, idx);
  if (t->tt_ == 0x45 &&
      (slot = ((lua_Unsigned)n - 1u < ((Table *)t->value_.gc)->alimit)
                ? &((Table *)t->value_.gc)->array[n - 1]
                : luaH_getint((Table *)t->value_.gc, n),
       (slot->tt_ & 0xf) != 0)) {
    TValue *s = (TValue *)slot;
    TValue *v = L->top - 1;
    s->value_ = v->value_;
    s->tt_ = v->tt_;
    if ((v->tt_ & 0x40) && (t->value_.gc->marked & 0x20) && (v->value_.gc->marked & 0x18))
      luaC_barrierback_(L, t->value_.gc);
  }
  else {
    TValue aux;
    setivalue(&aux, n);
    luaV_finishset(L, t, &aux, L->top - 1, slot);
  }
  L->top--;
}
