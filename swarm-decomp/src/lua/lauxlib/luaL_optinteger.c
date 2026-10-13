#include "lauxlib.h"

lua_Integer luaL_optinteger (lua_State *L, int arg, lua_Integer def) {
  return lua_isnoneornil(L, arg) ? def : luaL_checkinteger(L, arg);
}
