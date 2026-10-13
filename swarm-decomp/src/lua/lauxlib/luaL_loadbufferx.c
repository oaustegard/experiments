#include "LoadS.h"
#include "lua_api_headers.h"

int luaL_loadbufferx (lua_State *L, const char *buff, size_t size, const char *name, const char *mode) {
  LoadS ls;
  ls.s = buff;
  ls.size = size;
  return lua_load(L, getS, &ls, name, mode);
}
