#include "lstate.h"
#include "fn_luaH_get.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

int lua_rawget (lua_State *L, int idx) {
  Table *t;
  const TValue *val;
  Value v;
  t = (Table *)index2value(L, idx)->value_.gc;
  val = luaH_get(t, L->top - 1);
  v = val->value_;
  L->top--;
  if ((val->tt_ & 0xf) != 0) {
    L->top->value_ = v;
    L->top->tt_ = val->tt_;
    L->top++;
    return val->tt_ & 0xf;
  }
  L->top->tt_ = 0;
  L->top++;
  return 0;
}
