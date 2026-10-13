#include "lauxlib.h"
#include "lualib.h"

#define UTF8PATT	"[\0-\x7F\xC2-\xFD][\x80-\xBF]*"

static const luaL_Reg funcs[];

LUAMOD_API int luaopen_utf8 (lua_State *L) {
  luaL_newlib(L, funcs);
  lua_pushlstring(L, UTF8PATT, sizeof(UTF8PATT)/sizeof(char) - 1);
  lua_setfield(L, -2, "charpattern");
  return 1;
}
