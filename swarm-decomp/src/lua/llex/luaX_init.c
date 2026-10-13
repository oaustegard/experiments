#include "llex_macros.h"

void luaX_init(lua_State *L) {
  int i;
  TString *e = luaS_newlstr(L, "_ENV", sizeof("_ENV") - 1);
  luaC_fix(L, (GCObject *)e);
  for (i = 0; i < 22; i++) {
    TString *ts = luaS_new(L, luaX_tokens[i]);
    luaC_fix(L, (GCObject *)ts);
    ts->extra = (lu_byte)(i + 1);
  }
}
