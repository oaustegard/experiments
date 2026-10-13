#include "lstate.h"
#include "fn_luaC_barrierback_.h"
#include "fn_luaH_setint.h"
#include "lua.h"

LUA_API void lua_rawseti (lua_State *L, int idx, lua_Integer n) {
  Table *t;
  t = (Table *)index2value(L, idx)->value_.gc;
  luaH_setint(L, t, n, L->top - 1);
  if (((L->top - 1)->tt_ & 0x40) && (t->marked & 0x20) && ((L->top - 1)->value_.gc->marked & 0x18))
    luaC_barrierback_(L, (GCObject *)t);
  L->top--;
}
