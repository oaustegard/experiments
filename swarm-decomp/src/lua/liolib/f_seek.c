#include <errno.h>
#include <stdio.h>
#include "lauxlib.h"

static int f_seek (lua_State *L) {
  static const int mode[] = {SEEK_SET, SEEK_CUR, SEEK_END};
  static const char *const modenames[] = {"set", "cur", "end", NULL};
  FILE *f = tofile(L);
  int op = luaL_checkoption(L, 2, "cur", modenames);
  lua_Integer p3 = luaL_optinteger(L, 3, 0);
  off_t offset = (off_t)p3;
  errno = 0;
  op = fseeko(f, offset, mode[op]);
  if (__builtin_expect((op) != 0, 0))
    return luaL_fileresult(L, 0, NULL);
  else {
    lua_pushinteger(L, (lua_Integer)ftello(f));
    return 1;
  }
}
