#include "lstate.h"
#include "fn_luaH_getint.h"
#include "fn_luaV_finishget.h"
#include "lobject_tags.h"
#include "lua.h"

LUA_API int lua_geti (lua_State *L, int idx, lua_Integer n) {
  TValue *t;
  const TValue *slot = NULL;
  t = index2value(L, idx);
  if (t->tt_ == 0x45 &&
      (slot = ((lua_Unsigned)n - 1u < ((Table *)t->value_.gc)->alimit)
                ? &((Table *)t->value_.gc)->array[n - 1]
                : luaH_getint((Table *)t->value_.gc, n),
       (slot->tt_ & 0xf) != 0)) {
    L->top->value_ = slot->value_;
    L->top->tt_ = slot->tt_;
  }
  else {
    TValue aux;
    setivalue(&aux, n);
    luaV_finishget(L, t, &aux, L->top, slot);
  }
  L->top++;
  return L->top[-1].tt_ & 0xf;
}
