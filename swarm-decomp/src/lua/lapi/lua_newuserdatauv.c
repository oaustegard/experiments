#include <stddef.h>
#include "lstate.h"
#include "fn_luaC_step.h"
#include "fn_luaS_newudata.h"
#include "Udata.h"
#include "lua.h"

LUA_API void *lua_newuserdatauv (lua_State *L, size_t size, int nuvalue) {
  Udata *u;
  size_t off;
  u = luaS_newudata(L, size, nuvalue);
  L->top->value_.gc = (GCObject *)u;
  L->top->tt_ = 0x47;
  L->top++;
  if (G(L)->GCdebt > 0)
    luaC_step(L);
  off = (u->nuvalue == 0) ? 0x20 : offsetof(Udata, uv) + sizeof(UValue) * u->nuvalue;
  return (char *)u + off;
}
