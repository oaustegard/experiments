#include "lstate.h"
#include "lua.h"
#include "fn_luaE_warning.h"

LUA_API void lua_warning (lua_State *L, const char *msg, int tocont) {
  luaE_warning(L, msg, tocont);
}
