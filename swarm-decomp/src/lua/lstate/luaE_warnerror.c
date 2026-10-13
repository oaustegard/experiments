#include "lua.h"
#include "lstate.h"
#include "fn_luaE_warning.h"

void luaE_warnerror (lua_State *L, const char *where) {
  TValue *errobj = L->top - 1;
  const char *msg = ((errobj->tt_ & 0xf) == 4)
    ? ((TString *)errobj->value_.gc)->contents
    : "error object is not a string";
  luaE_warning(L, "error in ", 1);
  luaE_warning(L, where, 1);
  luaE_warning(L, " (", 1);
  luaE_warning(L, msg, 1);
  luaE_warning(L, ")", 0);
}
