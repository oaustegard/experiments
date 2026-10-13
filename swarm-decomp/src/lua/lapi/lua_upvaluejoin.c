#include "lstate.h"
#include "fn_luaC_barrier_.h"
#include "lua.h"

LUA_API void lua_upvaluejoin (lua_State *L, int fidx1, int n1, int fidx2, int n2) {
  LClosure *f1;
  UpVal **up1 = getupvalref(L, fidx1, n1, &f1);
  UpVal **up2 = getupvalref(L, fidx2, n2, NULL);
  *up1 = *up2;
  if ((((GCObject *)f1)->marked & 0x20) && (((GCObject *)*up1)->marked & 0x18))
    luaC_barrier_(L, (GCObject *)f1, (GCObject *)*up1);
}
