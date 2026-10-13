#include "lauxlib.h"
#include "lutf8lib_common.h"

static void pushutfchar (lua_State *L, int arg) {
  lua_Integer code = luaL_checkinteger(L, arg);
  luaL_argcheck(L, (lua_Unsigned)code <= MAXUTF, arg, "value out of range");
  lua_pushfstring(L, "%U", (long)code);
}
