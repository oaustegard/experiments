#include <string.h>
#include "lauxlib.h"
#include "prepbuffsize.h"

void luaL_addvalue (luaL_Buffer *B) {
  lua_State *L = B->L;
  size_t len;
  const char *s = lua_tolstring(L, -1, &len);
  char *b = prepbuffsize(B, len, -2);
  memcpy(b, s, len);
  B->n += len;
  lua_settop(L, -2);
}
