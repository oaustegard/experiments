#include "lstate.h"
#include "fn_luaH_set.h"
#include "fn_luaC_barrierback_.h"
#include "lua.h"

static TValue *index2value (lua_State *L, int idx);

LUA_API void lua_rawset (lua_State *L, int idx) {
  Table *t;
  StkId top = L->top;
  t = (Table *)index2value(L, idx)->value_.gc;
  luaH_set(L, t, top - 2, top - 1);
  t->flags &= ~0x3f;
  if ((L->top - 1)->tt_ & 0x40) {
    if ((t->marked & 0x20) && ((L->top - 1)->value_.gc->marked & 0x18))
      luaC_barrierback_(L, (GCObject *)t);
  }
  L->top -= 2;
}
