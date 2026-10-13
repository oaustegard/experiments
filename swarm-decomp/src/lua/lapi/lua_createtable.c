#include "lstate.h"
#include "fn_luaH_new.h"
#include "fn_luaH_resize.h"
#include "fn_luaC_step.h"
#include "lua.h"

LUA_API void lua_createtable (lua_State *L, int narray, int nrec) {
  Table *t;
  t = luaH_new(L);
  L->top->value_.gc = (GCObject *)t;
  L->top->tt_ = 5 | (1 << 6);
  L->top++;
  if (narray > 0 || nrec > 0)
    luaH_resize(L, t, narray, nrec);
  if (L->l_G->GCdebt > 0)
    luaC_step(L);
}
