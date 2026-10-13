#include "lstate.h"
#include "lua.h"

static const char *aux_upvalue (TValue *fi, int n, TValue **val, GCObject **owner);
TValue *index2value (lua_State *L, int idx);

LUA_API const char *lua_getupvalue (lua_State *L, int funcindex, int n) {
  const char *name;
  TValue *val = NULL;
  name = aux_upvalue(index2value(L, funcindex), n, &val, NULL);
  if (name) {
    setobj(L->top, val);
    L->top++;
  }
  return name;
}
