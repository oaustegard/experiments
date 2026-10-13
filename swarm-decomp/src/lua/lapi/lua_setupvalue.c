#include "lstate.h"
#include "fn_luaC_barrier_.h"
#include "lua.h"

static const char *aux_upvalue (TValue *fi, int n, TValue **val, GCObject **owner);

LUA_API const char *lua_setupvalue (lua_State *L, int funcindex, int n) {
  const char *name;
  TValue *val = NULL;
  GCObject *owner = NULL;
  TValue *fi;
  fi = index2value(L, funcindex);
  name = aux_upvalue(fi, n, &val, &owner);
  if (name) {
    L->top--;
    val->value_ = L->top->value_;
    val->tt_ = L->top->tt_;
    if ((val->tt_ & 0x40) && (owner->marked & 0x20) && (val->value_.gc->marked & 0x18))
      luaC_barrier_(L, owner, val->value_.gc);
  }
  return name;
}
