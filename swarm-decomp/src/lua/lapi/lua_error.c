#include "lstate.h"
#include "fn_luaD_throw.h"
#include "fn_luaG_errormsg.h"
#include "lua.h"

#define G(L) ((L)->l_G)

LUA_API int lua_error (lua_State *L) {
  TValue *errobj = L->top - 1;
  if (errobj->tt_ == (4 | (0 << 4) | (1 << 6)) &&
      errobj->value_.gc == (GCObject *)G(L)->memerrmsg)
    luaD_throw(L, 4);
  else
    luaG_errormsg(L);
}
