#include "lauxlib.h"

static int read_chars (lua_State *L, FILE *f, size_t n) {
  size_t nr;
  char *p;
  luaL_Buffer b;
  luaL_buffinit(L, &b);
  p = luaL_prepbuffsize(&b, n);
  nr = fread(p, sizeof(char), n, f);
  luaL_addsize(&b, nr);
  luaL_pushresult(&b);
  return (nr > 0);
}
