#include <locale.h>
#include "lauxlib.h"

static int os_setlocale (lua_State *L) {
  static const int cat[] = {LC_ALL, LC_COLLATE, LC_CTYPE, LC_MONETARY,
                            LC_NUMERIC, LC_TIME};
  static const char *const catnames[] = {"all", "collate", "ctype",
    "monetary", "numeric", "time", NULL};
  const char *l = luaL_optlstring(L, 1, NULL, NULL);
  int op = luaL_checkoption(L, 2, "all", catnames);
  lua_pushstring(L, setlocale(cat[op], l));
  return 1;
}
