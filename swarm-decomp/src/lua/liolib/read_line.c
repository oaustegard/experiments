#include "lauxlib.h"

static int read_line (lua_State *L, FILE *f, int chop) {
  luaL_Buffer b;
  int c;
  luaL_buffinit(L, &b);
  do {
    char *buff = luaL_prepbuffsize(&b, LUAL_BUFFERSIZE);
    int i = 0;
    flockfile(f);
    while (i < LUAL_BUFFERSIZE && (c = getc_unlocked(f)) != EOF && c != '\n')
      buff[i++] = c;
    funlockfile(f);
    luaL_addsize(&b, i);
  } while (c != EOF && c != '\n');
  if (!chop && c == '\n')
    luaL_addchar(&b, c);
  luaL_pushresult(&b);
  return (c == '\n' || lua_rawlen(L, -1) > 0);
}
