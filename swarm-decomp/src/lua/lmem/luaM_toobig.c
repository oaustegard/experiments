#include "fn_luaG_runerror.h"
#include "fn_luaM_toobig.h"

_Noreturn void luaM_toobig (lua_State *L) {
  luaG_runerror(L, "memory allocation error: block too big");
}
